import { initPhysics } from '../src/sim/physicsEngine';
import { createWorld, DEFAULT_SPEC, DEFAULT_ASSISTS } from '../src/sim/spawn';
import { step } from '../src/sim/world';
import { startMatch } from '../src/sim/match';
import { getObservationVector, applyAction, getReward, type RLRewardConfig } from '../src/sim/rl';
import type { World, RobotCommand, RobotSetup } from '../src/types';
import * as C from '../src/config';
import * as readline from 'readline';

async function main() {
  await initPhysics();

  let world: World | null = null;
  let rewardConfig: RLRewardConfig = {
    scoreWeight: 1.0,
    intakeWeight: 0.2,
    motifMatchWeight: 0.5,
    foulWeight: 2.0,
    timePenalty: 0.001,
    ballApproachWeight: 0.02,
    shootZoneApproachWeight: 0.02,
    intakeAlignmentWeight: 0.5,
    intakeAlignmentAbsWeight: 0.0, // default 0; overridden by Python config
    earlyGatePenalty: 0.0,         // default 0; overridden by Python config
    fullGateBonus: 0.0,            // default 0; overridden by Python config
    scoreCloseWeight: 0.0,         // default 0; overridden by Python config
    scoreFarWeight: 0.0,           // default 0; overridden by Python config
    overflowPenalty: 0.0,          // default 0; overridden by Python config
    gateApproachPenalty: 0.0,      // default 0; overridden by Python config
    autoOppositeSidePenalty: 0.0,  // default 0; overridden by Python config
  };

  const frameSkip = 8;
  const robotId = 0;

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });

  let stage = 2;

function getStage1SpawnPos(robotPos?: { x: number; y: number }): { x: number; y: number } {
  let x = 0, y = 0;
  for (let attempt = 0; attempt < 20; attempt++) {
    x = (Math.random() * 2 - 1) * 50;
    y = (Math.random() * 2 - 1) * 50;
    if (robotPos) {
      const d = Math.hypot(x - robotPos.x, y - robotPos.y);
      if (d > 20) break;
    } else {
      break;
    }
  }
  return { x, y };
}

  rl.on('line', (line) => {
    try {
      const msg = JSON.parse(line);

      if (msg.type === 'reset') {
        stage = msg.stage || 2;
        const setups: RobotSetup[] = msg.setups || [{
          id: robotId,
          alliance: msg.alliance || 'blue',
          spec: msg.spec || DEFAULT_SPEC,
          assists: msg.assists || DEFAULT_ASSISTS,
          startIndex: msg.startIndex !== undefined ? msg.startIndex : 0,
          autoPathEnabled: false,
        }];
        if (msg.rewardConfig) {
          rewardConfig = { ...rewardConfig, ...msg.rewardConfig };
        }

        // Deterministic seed if provided, otherwise random
        const seed = msg.seed !== undefined ? msg.seed : Math.floor(Math.random() * 100000);
        world = createWorld('match', seed, setups);
        startMatch(world);

        // STAGE 1 INITIALIZATION: Clear hoppers and relocate any preloaded/held balls to random ground positions
        if (stage === 1) {
          const mainRobot = world.robots[0];
          for (const b of world.balls) {
            if (b.state.kind === 'held') {
              b.state = { kind: 'ground' };
              b.pos = getStage1SpawnPos(mainRobot?.pos);
              b.vel = { x: 0, y: 0 };
              b.z = 0;
              b.vz = 0;
            }
          }
          for (const r of world.robots) {
            r.hopper = [];
          }
        }

        const obs = getObservationVector(world, robotId);
        const observations: { [id: number]: number[] } = {};
        for (const r of world.robots) {
          observations[r.id] = getObservationVector(world, r.id);
        }

        console.log(JSON.stringify({
          type: 'state',
          observation: obs,
          observations,
          reward: 0.0,
          terminated: false,
          info: {
            score: 0,
            phase: world.match.phase,
            timeLeft: world.match.phaseTimeLeft,
            robots: world.robots.map(r => ({
              id: r.id,
              alliance: r.alliance,
              pos: r.pos,
              heading: r.heading,
              spec: r.spec
            }))
          },
        }));

      } else if (msg.type === 'step') {
        if (!world) {
          console.log(JSON.stringify({ error: 'Environment not reset' }));
          return;
        }

        const prevWorld: World = JSON.parse(JSON.stringify(world));

        // Build commands for all active robots
        const commands = new Map<number, RobotCommand>();
        if (msg.actions) {
          for (const key of Object.keys(msg.actions)) {
            const rid = parseInt(key, 10);
            commands.set(rid, applyAction(msg.actions[key]));
          }
        } else {
          commands.set(robotId, applyAction(msg.action));
        }

        // Advance simulation by frameSkip ticks
        for (let i = 0; i < frameSkip; i++) {
          step(world, C.SIM_DT, commands);
        }

        const reward = getReward(world, prevWorld, robotId, rewardConfig);
        const terminated = world.match.phase === 'post';

        // STAGE 1 LOGIC: As soon as a ball is intaked, relocate it to a random ground position and clear hopper
        if (stage === 1) {
          const mainRobot = world.robots[0];
          for (const b of world.balls) {
            if (b.state.kind === 'held') {
              b.state = { kind: 'ground' };
              b.pos = getStage1SpawnPos(mainRobot?.pos);
              b.vel = { x: 0, y: 0 };
              b.z = 0;
              b.vz = 0;
            }
          }
          for (const r of world.robots) {
            r.hopper = [];
          }
        }

        const obs = getObservationVector(world, robotId);
        const observations: { [id: number]: number[] } = {};
        for (const r of world.robots) {
          observations[r.id] = getObservationVector(world, r.id);
        }

        console.log(JSON.stringify({
          type: 'state',
          observation: obs,
          observations,
          reward,
          terminated,
          info: {
            score: world.match.scores[world.robots[0].alliance].total,
            phase: world.match.phase,
            timeLeft: world.match.phaseTimeLeft,
            fouls: world.match.fouls[world.robots[0].alliance],
            robots: world.robots.map(r => ({
              id: r.id,
              alliance: r.alliance,
              pos: r.pos,
              heading: r.heading,
              spec: r.spec
            }))
          },
        }));

      } else {
        console.log(JSON.stringify({ error: `Unknown message type: ${msg.type}` }));
      }
    } catch (e: any) {
      console.log(JSON.stringify({ error: `Exception: ${e.message}` }));
    }
  });
}

main().catch((err) => {
  console.error("Fatal initialization error:", err);
  process.exit(1);
});
