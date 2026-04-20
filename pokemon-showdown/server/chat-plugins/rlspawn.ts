// rlspawn.ts

import type {Room} from "../rooms";
import {Rooms} from "../rooms";
import {Users} from "../users";

// --- HELPERS ---

function toID(text: any): string {
  if (text === null || text === undefined) return "";
  return ("" + text).toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function getUser(name: string) {
  const U: any = Users as any;
  return (U.getExact?.(name) ?? U.get?.(name) ?? null);
}

function roomPlayerNames(room: any): [string | null, string | null] {
  const battle = room?.battle;
  if (!battle) return [null, null];
  const p1 = battle.p1?.userid || battle.p1?.user?.userid || battle.p1?.name || battle.p1?.user?.name;
  const p2 = battle.p2?.userid || battle.p2?.user?.userid || battle.p2?.name || battle.p2?.user?.name;
  return [p1 ? String(p1) : null, p2 ? String(p2) : null];
}

function getBattleCountBetween(u1: any, u2: any): number {
  const x = toID(u1?.userid || u1?.name);
  const y = toID(u2?.userid || u2?.name);
  if (!x || !y) return 0;

  const R: any = Rooms as any;
  const rooms: Map<string, Room> = (R.rooms as any) || new Map();
  let n = 0;
  
  for (const room of rooms.values()) {
    const battle: any = (room as any).battle;
    if (!battle || battle.ended) continue;

    const [p1, p2] = roomPlayerNames(room as any);
    if (!p1 || !p2) continue;

    const a = toID(p1);
    const b = toID(p2);
    if ((a === x && b === y) || (a === y && b === x)) n++;
  }
  return n;
}

// --- CORE LOGIC ---

/**
 * Creates a battle and stamps it with a Birth Certificate (_rlBorn).
 */
function createBattleRoom(format: string, u1: any, u2: any) {
  const R: any = Rooms as any;
  const formatid = toID(format);
  let room: any = null;

  // Attempt creation using various API signatures
  if (typeof R.createBattle === "function") {
    try {
      room = R.createBattle({ format: formatid, rated: false, players: [{user: u1, team: ""}, {user: u2, team: ""}] });
    } catch {}
  }
  
  if (!room && typeof R.createBattleRoom === "function") {
    try {
      room = R.createBattleRoom({ format: formatid, rated: false, players: [{user: u1, team: ""}, {user: u2, team: ""}] });
    } catch {}
  }

  if (room) {
    // [!] BIRTH CERTIFICATE: Manually track creation time because server doesn't
    (room as any)._rlBorn = Date.now();
    try { u1.joinRoom(room); } catch {}
    try { u2.joinRoom(room); } catch {}
  } else {
    throw new Error("Battle creation API failed.");
  }
  
  return room;
}

// --- COMMANDS ---

type Key = string;
const sched = new Map<Key, any>();

function key(u1: string, u2: string, format: string) {
  return `${String(u1).toLowerCase()}|${String(u2).toLowerCase()}|${toID(format)}`;
}

export const commands: Chat.ChatCommands = {
  
  /**
   * HIGH DENSITY RECONCILIATION
   * Returns a raw CSV string of battle IDs.
   * Compactness allows sending ~4000 IDs in a single packet without pagination.
   */
  rlactive(target, room, user) {
    const R: any = Rooms as any;
    const rooms: Map<string, Room> = (R.rooms as any) || new Map();
    const userId = user.id;
    
    // Use a pre-allocated array for performance
    const battles: string[] = [];

    for (const r of rooms.values()) {
        const battle: any = (r as any).battle;
        if (!battle || battle.ended) continue;

        const [p1, p2] = roomPlayerNames(r);
        if (toID(p1) === userId || toID(p2) === userId) {
            battles.push(r.roomid);
        }
    }
    
    // Optimization: Join with commas. No JSON overhead.
    // Estimated size for 4096 battles: ~60KB. Safe for single frame.
    const payload = battles.join(",");
    
    user.send(`|queryresponse|rlactive|${payload}`);
    // console.log(`[rlactive] Sent ${battles.length} IDs to ${user.name} (${(payload.length/1024).toFixed(1)} KB)`);
  },

  rlrescue(target, room, user) {
    const targetRoomId = toID(target);
    
    // [!] LOG IMMEDIATELY
    // This proves the packet reached the server.
    // console.log(`[rlrescue] Received request for ${targetRoomId} from ${user.name}`);

    const R: any = Rooms as any;
    const targetRoom = R.get(targetRoomId);

    if (!targetRoom) {
        // console.log(`[rlrescue] Aborting: Room ${targetRoomId} does not exist.`);
        return;
    }

    if (targetRoom.battle) {
        const battle = targetRoom.battle;
        const p1 = battle.p1?.userid || battle.p1?.user?.userid;
        const p2 = battle.p2?.userid || battle.p2?.user?.userid;
        
        if (user.id !== p1 && user.id !== p2) {
             // console.log(`[rlrescue] Aborting: User ${user.id} is not a player in ${targetRoomId}.`);
             return;
        }
    }

    try {
        // console.log(`[rlrescue] NUKE EXECUTE: Expiring ${targetRoomId}`);
        targetRoom.expire(); 
    } catch (e: any) {
        console.error(`[rlrescue] Expire failed for ${targetRoomId}:`, e.message);
        if (targetRoom.destroy) targetRoom.destroy();
    }
  },
  
  rlautospawn(target, room, user) {
    const parts = target.split(",").map(s => s.trim()).filter(Boolean);
    if (parts.length < 4) return console.error("[rlautospawn] Invalid args");

    const [u1n, u2n, formatRaw, targetStr] = parts;
    const format = toID(formatRaw);
    const tgt = Math.max(1, Math.min(4096, parseInt(targetStr, 10)));

    const u1 = getUser(u1n);
    const u2 = getUser(u2n);
    if (!u1 || !u2) return console.error("[rlautospawn] Users offline");

    const k = key(u1n, u2n, format);
    if (sched.has(k)) {
        sched.get(k).target = tgt;
        return;
    }

    // Tick logic
    const tick = () => {
      const a = getUser(u1n);
      const b = getUser(u2n);
      if (!a || !b) return;
      
      const cur = getBattleCountBetween(a, b);
      const need = tgt - cur;
      
      if (need <= 0) return;
      
      // Burst limit: spawn max 50 at a time
      const burst = Math.min(need, 50);
      for (let i = 0; i < burst; i++) {
        try {
          createBattleRoom(format, a, b);
        } catch (e: any) {
          console.error("[rlautospawn] Create error:", e.message);
          return;
        }
      }
    };

    const state = {
      target: tgt,
      timer: setInterval(tick, 50), // Check every 50ms
    };

    sched.set(k, state);
    // console.log(`[rlautospawn] Started ${u1n} vs ${u2n} (${tgt})`);
  },

  rlautooff(target, room, user) {
    const parts = target.split(",").map(s => s.trim()).filter(Boolean);
    if (parts.length < 3) return;
    const k = key(parts[0], parts[1], parts[2]);
    const s = sched.get(k);
    if (s) {
        clearInterval(s.timer);
        sched.delete(k);
        // console.log(`[rlautospawn] Stopped ${parts[0]} vs ${parts[1]}`);
    }
  },
};

// --- GARBAGE COLLECTION ---

// Maximum lifespan of an RL room in milliseconds (e.g., 10 minutes)
// Adjust this based on whether you are doing Imitation (fast) or PPO (slower)
const RL_MAX_LIFESPAN_MS = 5 * 60 * 1000; 

setInterval(() => {
  const R: any = Rooms as any;
  const rooms: Map<string, Room> = (R.rooms as any) || new Map();
  const now = Date.now();
  let reaped = 0;

  for (const room of rooms.values()) {
    const rlBorn = (room as any)._rlBorn;
    
    // Only target rooms created by our autospawner
    if (rlBorn && (now - rlBorn > RL_MAX_LIFESPAN_MS)) {
      const battle: any = (room as any).battle;
      
      // If the battle exists and hasn't natively ended, nuke it
      if (battle && !battle.ended) {
        try {
          (room as any).expire();
          reaped++;
        } catch (e: any) {
          if ((room as any).destroy) (room as any).destroy();
          reaped++;
        }
      }
    }
  }

  // Optional: Uncomment to monitor server-side GC activity
  // if (reaped > 0) {
  //   console.log(`[rlspawn GC] Reaped ${reaped} orphaned battle rooms.`);
  // }

}, 60 * 1000); // Run the sweeper every 60 seconds