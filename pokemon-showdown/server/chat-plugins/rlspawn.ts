/**
 * rlspawn.ts
 * * Custom Pokémon Showdown chat plugin to support high-throughput, 
 * automated Reinforcement Learning (RL) self-play.
 */

import type { Room } from "../rooms";
import { Rooms } from "../rooms";
import { Users } from "../users";

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Normalizes text to a standard Pokémon Showdown ID (lowercase, alphanumeric).
 */
function toID(text: any): string {
    if (text === null || text === undefined) return "";
    return ("" + text).toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/**
 * Safely fetches a User object by name, falling back to substring matching if needed.
 */
function getUser(name: string) {
    const U: any = Users as any;
    return (U.getExact?.(name) ?? U.get?.(name) ?? null);
}

/**
 * Extracts the user IDs of both players in a given battle room.
 */
function roomPlayerNames(room: any): [string | null, string | null] {
    const battle = room?.battle;
    if (!battle) return [null, null];

    const p1 = battle.p1?.userid || battle.p1?.user?.userid || battle.p1?.name || battle.p1?.user?.name;
    const p2 = battle.p2?.userid || battle.p2?.user?.userid || battle.p2?.name || battle.p2?.user?.name;
    
    return [p1 ? String(p1) : null, p2 ? String(p2) : null];
}

/**
 * Scans all active rooms to count how many ongoing battles exist between two specific users.
 * Note: This is an O(N) operation over all rooms.
 */
function getBattleCountBetween(u1: any, u2: any): number {
    const user1Id = toID(u1?.userid || u1?.name);
    const user2Id = toID(u2?.userid || u2?.name);
    if (!user1Id || !user2Id) return 0;

    const R: any = Rooms as any;
    const rooms: Map<string, Room> = (R.rooms as any) || new Map();
    let count = 0;
    
    for (const room of rooms.values()) {
        const battle: any = (room as any).battle;
        if (!battle || battle.ended) continue;

        const [p1, p2] = roomPlayerNames(room);
        if (!p1 || !p2) continue;

        const p1Id = toID(p1);
        const p2Id = toID(p2);
        
        if ((p1Id === user1Id && p2Id === user2Id) || (p1Id === user2Id && p2Id === user1Id)) {
            count++;
        }
    }
    return count;
}

// ============================================================================
// CORE BATTLE CREATION
// ============================================================================

/**
 * Attempts to create a battle room using available server APIs and tags it.
 * * @throws {Error} If the battle creation API fails or is unavailable.
 */
function createBattleRoom(format: string, u1: any, u2: any): any {
    const R: any = Rooms as any;
    const formatid = toID(format);
    let room: any = null;

    // Gracefully handle different versions of the Pokémon Showdown API
    if (typeof R.createBattle === "function") {
        try {
            room = R.createBattle({ format: formatid, rated: false, players: [{user: u1, team: ""}, {user: u2, team: ""}] });
        } catch (e) {
            // Fallthrough to next method
        }
    }
    
    if (!room && typeof R.createBattleRoom === "function") {
        try {
            room = R.createBattleRoom({ format: formatid, rated: false, players: [{user: u1, team: ""}, {user: u2, team: ""}] });
        } catch (e) {
            // Fallthrough to error
        }
    }

    if (room) {
        // [!] BIRTH CERTIFICATE: Tag the room for custom garbage collection tracking
        (room as any)._rlBorn = Date.now();
        return room;
    }
    
    throw new Error(`Battle creation failed for format ${formatid}. Ensure the format exists and players are eligible.`);
}

// ============================================================================
// CHAT COMMANDS
// ============================================================================

// Scheduler state for autospawn loops
interface AutospawnState {
    target: number;
    timer: NodeJS.Timeout;
}

const sched = new Map<string, AutospawnState>();

/**
 * Generates a unique key for the autospawn scheduler.
 */
function getSchedulerKey(u1: string, u2: string, format: string): string {
    return `${toID(u1)}|${toID(u2)}|${toID(format)}`;
}

export const commands: Chat.ChatCommands = {
    
    /**
     * HIGH DENSITY RECONCILIATION
     * Returns a compact, comma-separated list of active battle IDs for the requesting user.
     * Prevents JSON serialization overhead, allowing thousands of IDs in a single frame.
     */
    rlactive(target, room, user) {
        const R: any = Rooms as any;
        const rooms: Map<string, Room> = (R.rooms as any) || new Map();
        const userId = user.id;
        
        const activeBattles: string[] = [];

        for (const r of rooms.values()) {
            const battle: any = (r as any).battle;
            if (!battle || battle.ended) continue;

            const [p1, p2] = roomPlayerNames(r);
            if (toID(p1) === userId || toID(p2) === userId) {
                activeBattles.push(r.roomid);
            }
        }
        
        const payload = activeBattles.join(",");
        user.send(`|queryresponse|rlactive|${payload}`);
    },

    /**
     * DEADLOCK RESOLUTION
     * Forcibly expires a stalled battle room. Requires the requester to be a player in that room.
     */
    rlrescue(target, room, user) {
        const targetRoomId = toID(target);
        if (!targetRoomId) return;

        const R: any = Rooms as any;
        const targetRoom = R.get(targetRoomId);

        if (!targetRoom) return; // Silently abort if room doesn't exist

        // Security check: Ensure the user is actually participating in the target battle
        if (targetRoom.battle) {
            const battle = targetRoom.battle;
            const p1 = battle.p1?.userid || battle.p1?.user?.userid;
            const p2 = battle.p2?.userid || battle.p2?.user?.userid;
            
            if (user.id !== p1 && user.id !== p2) {
                 return this.errorReply(`Access denied: You are not a player in ${targetRoomId}.`);
            }
        }

        try {
            targetRoom.expire(); 
        } catch (e: any) {
            // Fallback destruction if graceful expiration fails
            if (typeof targetRoom.destroy === 'function') {
                targetRoom.destroy();
            }
        }
    },
    
    /**
     * TARGETED CONCURRENCY
     * Starts a background loop that maintains a specific number of concurrent battles between two users.
     * Usage: /rlautospawn [user1], [user2], [format], [target_count]
     */
    rlautospawn(target, room, user) {
        const parts = target.split(",").map(s => s.trim()).filter(Boolean);
        if (parts.length < 4) {
            return this.errorReply("Usage: /rlautospawn [user1], [user2], [format], [target_count]");
        }

        const [u1n, u2n, formatRaw, targetStr] = parts;
        const format = toID(formatRaw);
        const targetCount = Math.max(1, Math.min(4096, parseInt(targetStr, 10)));

        const u1 = getUser(u1n);
        const u2 = getUser(u2n);
        if (!u1 || !u2) {
            return this.errorReply("One or both users are offline or do not exist.");
        }

        const schedKey = getSchedulerKey(u1n, u2n, format);
        
        // Update target if loop is already running
        if (sched.has(schedKey)) {
            sched.get(schedKey)!.target = targetCount;
            return;
        }

        // The maintenance loop
        const tick = () => {
            const a = getUser(u1n);
            const b = getUser(u2n);
            if (!a || !b) return; // Pause spawning if a user drops offline
            
            const currentCount = getBattleCountBetween(a, b);
            const deficit = targetCount - currentCount;
            
            if (deficit <= 0) return;
            
            // Burst limit: Prevents event loop starvation by capping spawns per tick
            const burstCount = Math.min(deficit, 50);
            
            for (let i = 0; i < burstCount; i++) {
                try {
                    createBattleRoom(format, a, b);
                } catch (e: any) {
                    // Stop the loop if creation fundamentally fails (e.g., bad format)
                    clearInterval(state.timer);
                    sched.delete(schedKey);
                    return;
                }
            }
        };

        const state: AutospawnState = {
            target: targetCount,
            timer: setInterval(tick, 50), // 50ms interval provides smooth, high-throughput scaling
        };

        sched.set(schedKey, state);
    },

    /**
     * GRACEFUL SHUTDOWN
     * Stops an active autospawn loop.
     * Usage: /rlautooff [user1], [user2], [format]
     */
    rlautooff(target, room, user) {
        const parts = target.split(",").map(s => s.trim()).filter(Boolean);
        if (parts.length < 3) {
            return this.errorReply("Usage: /rlautooff [user1], [user2], [format]");
        }

        const schedKey = getSchedulerKey(parts[0], parts[1], parts[2]);
        const state = sched.get(schedKey);
        
        if (state) {
            clearInterval(state.timer);
            sched.delete(schedKey);
        }
    },
};