import type {Room} from "../rooms";
import {Rooms} from "../rooms";
import {Users} from "../users";

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

  // Prefer userid; fall back to name. Normalize with toID.
  const p1 = battle.p1?.userid || battle.p1?.user?.userid || battle.p1?.name || battle.p1?.user?.name;
  const p2 = battle.p2?.userid || battle.p2?.user?.userid || battle.p2?.name || battle.p2?.user?.name;
  return [p1 ? String(p1) : null, p2 ? String(p2) : null];
}

function countBattleRooms(): number {
  const R: any = Rooms as any;
  const rooms: Map<string, Room> = (R.rooms as any) || new Map();
  let n = 0;
  for (const room of rooms.values()) {
    const battle: any = (room as any).battle;
    if (!battle) continue;
    if (battle.ended) continue;
    n++;
  }
  return n;
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
    if (!battle) continue;

    // If PS keeps ended battles around briefly, this reduces over-counting.
    if (battle.ended) continue;

    const [p1, p2] = roomPlayerNames(room as any);
    if (!p1 || !p2) continue;

    const a = toID(p1);
    const b = toID(p2);

    if ((a === x && b === y) || (a === y && b === x)) n++;
  }
  return n;
}


/**
 * Modern PS (current master) uses Rooms.createBattle({...}) with an options object:
 * {format, players, rated, ...}
 * This script prefers that API and only falls back if absolutely necessary.
 */
function createBattleRoom(format: string, u1: any, u2: any) {
  const R: any = Rooms as any;
  const formatid = toID(format);

  if (typeof R.createBattle === "function") {
    // Preferred: players are objects (user + team). Team can be '' for randombattle.
    // This avoids failures where internal code does `players.map(...)` expecting objects.
    const optsObjPlayers = {
      format: formatid,
      rated: false,
      players: [
        {user: u1, team: ""},
        {user: u2, team: ""},
      ],
    };

    // Some forks/older builds accept players as raw Users.
    const optsRawPlayers = {
      format: formatid,
      rated: false,
      players: [u1, u2],
    };

    // Some internal tools pass `inputLog` (seen in server command usage patterns).
    const optsObjPlayersWithLog = {
      ...optsObjPlayers,
      inputLog: "",
    };

    try {
      return R.createBattle(optsObjPlayers);
    } catch {}
    try {
      return R.createBattle(optsObjPlayersWithLog);
    } catch {}
    try {
      return R.createBattle(optsRawPlayers);
    } catch {}
  }

  // Older/alternate method name on some builds
  if (typeof R.createBattleRoom === "function") {
    const opts = {
      format: formatid,
      rated: false,
      players: [
        {user: u1, team: ""},
        {user: u2, team: ""},
      ],
    };
    try {
      return R.createBattleRoom(opts);
    } catch {}
  }

  throw new Error("Battle creation API not found/supported on this PS build (expected Rooms.createBattle({...})).");
}

type Key = string;


const sched = new Map<
  Key,
  {
    u1: string;
    u2: string;
    format: string;
    target: number;
    timer: NodeJS.Timeout;
    // soft rate-limit so one broken state doesn't spam logs
    hadCreateError: boolean;
  }
>();

function key(u1: string, u2: string, format: string) {
  return `${String(u1).toLowerCase()}|${String(u2).toLowerCase()}|${toID(format)}`;
}

export const commands: Chat.ChatCommands = {
  rlautospawn(target, room, user) {
    // Usage: /rlautospawn userA, userB, gen9randombattle, 64
    // IMPORTANT: DO NOT send replies (poke-env crashes on non-| protocol lines).
    // We only log to server console.

    const parts = target.split(",").map(s => s.trim()).filter(Boolean);
    if (parts.length < 4) {
      // eslint-disable-next-line no-console
      console.error("[rlautospawn] Usage: /rlautospawn userA, userB, format, targetBattles");
      return;
    }

    const [u1n, u2n, formatRaw, targetStr] = parts;
    const format = toID(formatRaw);

    const tgtRaw = parseInt(targetStr, 10);
    if (!Number.isFinite(tgtRaw) || tgtRaw < 1) {
      // eslint-disable-next-line no-console
      console.error("[rlautospawn] targetBattles must be an integer >= 1");
      return;
    }
    const tgt = Math.max(1, Math.min(4096, tgtRaw));

    const u1 = getUser(u1n);
    const u2 = getUser(u2n);
    if (!u1 || !u2) {
      // eslint-disable-next-line no-console
      console.error("[rlautospawn] Both users must be online.", {u1n, u2n});
      return;
    }

    const k = key(u1n, u2n, format);
    if (sched.has(k)) {
      // eslint-disable-next-line no-console
      console.error("[rlautospawn] Already scheduled; use /rlautooff first.", k);
      return;
    }

    const state = {
      u1: u1n,
      u2: u2n,
      format,
      target: tgt,
      timer: null as any,
      hadCreateError: false,
    };

    // Tuneables:
    const TICK_MS = 200;
    const MAX_BURST_PER_TICK = 8;

    let lastLog = 0;
    let lastErrorAt = 0;
    
    const tick = () => {
      const a = getUser(u1n);
      const b = getUser(u2n);
      if (!a || !b) return;
    
      const cur = getBattleCountBetween(a, b);
      const need = tgt - cur;
    
      const now = Date.now();
      if (now - lastLog > 2000) {
        lastLog = now;
        // console.error so it shows even if stdout is busy
        console.error(
          `[rlautospawn] ${u1n} vs ${u2n} cur=${cur} need=${need} tgt=${tgt} totalBattleRooms=${countBattleRooms()}`
        );
      }
    
      if (need <= 0) return;
    
      // Backoff for 2s after any create error so we don't spam-fail.
      if (now - lastErrorAt < 2000) return;
    
      const burst = Math.min(need, MAX_BURST_PER_TICK);
      for (let i = 0; i < burst; i++) {
        try {
          createBattleRoom(format, a, b);
          // IMPORTANT: allow future errors to print again once we recover
          state.hadCreateError = false;
        } catch (e: any) {
          lastErrorAt = Date.now();
          if (!state.hadCreateError) {
            state.hadCreateError = true;
            console.error("[rlautospawn] createBattle failed:", e?.stack || e?.message || e);
          }
          return;
        }
      }
    };


    state.timer = setInterval(tick, TICK_MS);
    sched.set(k, state);
    tick();

    // eslint-disable-next-line no-console
    console.log(`[rlautospawn] started ${u1n} vs ${u2n} format=${format} target=${tgt}`);
  },

  rlautooff(target, room, user) {
    // Usage: /rlautooff userA, userB, gen9randombattle
    const parts = target.split(",").map(s => s.trim()).filter(Boolean);
    if (parts.length < 3) return;

    const [u1n, u2n, formatRaw] = parts;
    const format = toID(formatRaw);

    const k = key(u1n, u2n, format);
    const s = sched.get(k);
    if (!s) return;

    clearInterval(s.timer);
    sched.delete(k);

    // eslint-disable-next-line no-console
    console.log(`[rlautospawn] stopped ${u1n} vs ${u2n} format=${format}`);
  },
};
