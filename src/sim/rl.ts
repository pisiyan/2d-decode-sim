import type { RobotCommand, World } from '../types';
import * as C from '../config';
import { rot, dist } from '../math';
import { goalCenter, gateArmRect, inLaunchZone, launchTriangles, goalSide } from './field';

export interface RLRewardConfig {
  scoreWeight: number;
  intakeWeight: number;
  motifMatchWeight: number;
  foulWeight: number;
  timePenalty: number;
  ballApproachWeight: number;
  shootZoneApproachWeight: number;
  intakeAlignmentWeight: number;
  intakeAlignmentAbsWeight: number; // Absolute reward for currently facing nearby balls
  earlyGatePenalty: number;         // Penalty for opening gate with < 7 balls on rail
  fullGateBonus: number;            // Bonus per ball above 7 when opening gate
  scoreCloseWeight: number;         // Reward weight for classified balls shot from close
  scoreFarWeight: number;           // Reward weight for classified balls shot from far
  overflowPenalty: number;          // Penalty weight for overflowed balls
  gateApproachPenalty: number;      // Penalty weight for approaching the gate when < 9 balls in classifier
  autoOppositeSidePenalty: number;  // Penalty weight for going into the opposite side during auto
}

function distToSegment(p: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }): number {
  const l2 = dist(a, b) ** 2;
  if (l2 === 0) return dist(p, a);
  let t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
  t = Math.max(0, Math.min(1, t));
  return dist(p, {
    x: a.x + t * (b.x - a.x),
    y: a.y + t * (b.y - a.y)
  });
}

export function distanceToLaunchZone(p: { x: number; y: number }, alliance: any, closeOnly: boolean = false): number {
  if (closeOnly) {
    if (p.y >= Math.abs(p.x)) { // in main (close) launch zone
      return 0;
    }
    const f = C.FIELD_HALF;
    const mainTriangle = [{ x: 0, y: 0 }, { x: -f, y: f }, { x: f, y: f }];
    let minDist = Infinity;
    for (let i = 0; i < 3; i++) {
      const d = distToSegment(p, mainTriangle[i], mainTriangle[(i + 1) % 3]);
      if (d < minDist) {
        minDist = d;
      }
    }
    return minDist;
  }

  if (inLaunchZone(p, alliance)) {
    return 0;
  }
  const triangles = launchTriangles();
  let minDist = Infinity;
  for (const t of triangles) {
    for (let i = 0; i < 3; i++) {
      const d = distToSegment(p, t[i], t[(i + 1) % 3]);
      if (d < minDist) {
        minDist = d;
      }
    }
  }
  return minDist;
}

/** Formulates and normalizes the 45-dimensional state representation for the RL policy.
 * Symmetrical and local to the robot's heading to make it easy for an MLP to learn. */
export function getObservationVector(world: World, robotId: number): number[] {
  const r = world.robots.find((rr) => rr.id === robotId);
  if (!r) {
    return new Array(122).fill(0);
  }

  const obs: number[] = [];

  // 1. Robot Kinematics (9 dimensions)
  obs.push(r.pos.x / C.FIELD_HALF); // x relative to center [-1, 1]
  obs.push(r.pos.y / C.FIELD_HALF); // y relative to center [-1, 1]
  obs.push(Math.cos(r.heading));    // cos of heading
  obs.push(Math.sin(r.heading));    // sin of heading
  obs.push(r.vel.x / 150.0);        // vx normalized (typical max speed is ~120-150 in/s)
  obs.push(r.vel.y / 150.0);        // vy normalized
  obs.push(r.angVel / 5.0);         // angVel normalized
  obs.push(r.powerDraw / C.POWER_DRAW_MAX); // normalized current power draw
  const fireCooldown = r.fireReadyAt > world.time ? (r.fireReadyAt - world.time) : 0.0;
  obs.push(Math.min(fireCooldown / 1.0, 1.0)); // shooter ready cooldown fraction

  // 2. Hopper State (5 dimensions)
  obs.push(r.hopper.length / 3.0);  // hopper size normalized [0, 1]
  // 3 slots (0 = empty, -1 = purple, 1 = green)
  for (let i = 0; i < 3; i++) {
    if (i < r.hopper.length) {
      obs.push(r.hopper[i] === 'green' ? 1.0 : -1.0);
    } else {
      obs.push(0.0);
    }
  }
  obs.push(r.flywheelSpin); // flywheel spin level [0, 1]

  // 3. Match Info (9 dimensions)
  obs.push(world.match.phaseTimeLeft / 120.0); // normalized time left in phase [0, 1]
  
  // one-hot phase (pre, auto, transition, teleop, post, freeplay)
  const phases = ['pre', 'auto', 'transition', 'teleop', 'post', 'freeplay'];
  const phaseIdx = phases.indexOf(world.match.phase);
  for (let i = 0; i < 5; i++) {
    obs.push(phaseIdx === i ? 1.0 : 0.0);
  }
  
  // motif: 3 slots (-1 = purple, 1 = green, 0 = empty)
  for (let i = 0; i < 3; i++) {
    if (i < world.motif.length) {
      obs.push(world.motif[i] === 'green' ? 1.0 : -1.0);
    } else {
      obs.push(0.0);
    }
  }

  // 4. Goals relative to Robot (4 dimensions)
  const ownGoal = goalCenter(r.alliance);
  const ownGoalLocal = rot({ x: ownGoal.x - r.pos.x, y: ownGoal.y - r.pos.y }, -r.heading);
  obs.push(ownGoalLocal.x / (C.FIELD_HALF * 2));
  obs.push(ownGoalLocal.y / (C.FIELD_HALF * 2));

  const oppAlliance = r.alliance === 'red' ? 'blue' : 'red';
  const oppGoal = goalCenter(oppAlliance);
  const oppGoalLocal = rot({ x: oppGoal.x - r.pos.x, y: oppGoal.y - r.pos.y }, -r.heading);
  obs.push(oppGoalLocal.x / (C.FIELD_HALF * 2));
  obs.push(oppGoalLocal.y / (C.FIELD_HALF * 2));

  // 5. Gate relative to Robot (3 dimensions)
  const gateRect = gateArmRect(r.alliance);
  const gateCenterPos = { x: (gateRect.x0 + gateRect.x1) / 2, y: (gateRect.y0 + gateRect.y1) / 2 };
  const gateLocal = rot({ x: gateCenterPos.x - r.pos.x, y: gateCenterPos.y - r.pos.y }, -r.heading);
  obs.push(gateLocal.x / (C.FIELD_HALF * 2));
  obs.push(gateLocal.y / (C.FIELD_HALF * 2));
  
  const goalState = world.goals[r.alliance];
  obs.push(goalState.gatePos); // physical open fraction [0, 1]

  // Classifier rail count features (2 dimensions)
  // own alliance rail classified (non-overflow) balls count, normalized by max capacity 9
  const ownClassifierCount = world.balls.filter(
    (b) => b.state.kind === 'rail' && b.state.goal === r.alliance && !b.state.overflow
  ).length;
  obs.push(ownClassifierCount / 9.0);

  // opponent alliance rail classified balls count, normalized by max capacity 9
  const oppClassifierCount = world.balls.filter(
    (b) => b.state.kind === 'rail' && b.state.goal === oppAlliance && !b.state.overflow
  ).length;
  obs.push(oppClassifierCount / 9.0);

  // 6. Active Ground Balls Sorted by Target Priority (30 slots * 3 = 90 dimensions, total obs length = 122)
  // Filters ALL active ground balls (so relocated/respawned balls are 100% visible regardless of ID).
  // Uses heading-weighted priority scoring so Slot 0 ALWAYS contains the target ground ball right in front of the robot.
  const groundBalls = world.balls.filter((b) => b.state.kind === 'ground');
  const sortedGround = groundBalls.sort((a, b) => {
    const da = dist(a.pos, r.pos);
    const db = dist(b.pos, r.pos);
    
    // Relative angle offset to robot heading
    const angleA = Math.atan2(a.pos.y - r.pos.y, a.pos.x - r.pos.x) - r.heading;
    const angleB = Math.atan2(b.pos.y - r.pos.y, b.pos.x - r.pos.x) - r.heading;
    
    const normAngleA = Math.abs(Math.atan2(Math.sin(angleA), Math.cos(angleA)));
    const normAngleB = Math.abs(Math.atan2(Math.sin(angleB), Math.cos(angleB)));

    // Priority score: favors closer balls that the robot is facing (prevents slot thrashing & oscillation)
    const scoreA = da + normAngleA * 20.0;
    const scoreB = db + normAngleB * 20.0;

    return scoreA - scoreB;
  });

  const maxSlots = 30;
  for (let i = 0; i < maxSlots; i++) {
    if (i < sortedGround.length) {
      const b = sortedGround[i];
      const localBall = rot({ x: b.pos.x - r.pos.x, y: b.pos.y - r.pos.y }, -r.heading);
      obs.push(localBall.x / (C.FIELD_HALF * 2));
      obs.push(localBall.y / (C.FIELD_HALF * 2));
      obs.push(b.color === 'green' ? 1.0 : -1.0);
    } else {
      obs.push(0.0);
      obs.push(0.0);
      obs.push(0.0);
    }
  }

  return obs;
}

/** Translates flat model actions [-1, 1] into a RobotCommand. */
export function applyAction(action: number[]): RobotCommand {
  return {
    driveX: Math.max(-1.0, Math.min(1.0, action[0])),
    driveY: Math.max(-1.0, Math.min(1.0, action[1])),
    rotate: Math.max(-1.0, Math.min(1.0, action[2])),
    leftDrive: 0,
    rightDrive: 0,
    intake: false,
    fire: false,
  };
}

/** Computes the shaped reward for the step. */
export function getReward(
  world: World,
  prevWorld: World,
  robotId: number,
  config: RLRewardConfig
): number {
  const r = world.robots.find((rr) => rr.id === robotId);
  const prevR = prevWorld.robots.find((rr) => rr.id === robotId);
  if (!r || !prevR) return 0;

  let reward = 0;

  // 1. Scoring Reward: pattern matches, fouls, and leave points use general scoreWeight
  const getNonBallScore = (w: World, alliance: 'red' | 'blue') => {
    const s = w.match.scores[alliance];
    return s.leave + s.autoPattern + s.telePattern + s.foulPoints;
  };
  const generalScoreDiff = getNonBallScore(world, r.alliance) - getNonBallScore(prevWorld, r.alliance);
  reward += generalScoreDiff * config.scoreWeight;

  // 1b. Ball Classification and Overflow Rewards (shot zone dependent)
  const prevCommitted = prevWorld.balls.filter(
    (b) => b.state.kind === 'rail' && b.state.goal === r.alliance && !b.state.pending
  );
  const currCommitted = world.balls.filter(
    (b) => b.state.kind === 'rail' && b.state.goal === r.alliance && !b.state.pending
  );

  const newlyCommitted = currCommitted.filter((b) => !prevCommitted.some((pb) => pb.id === b.id));
  for (const b of newlyCommitted) {
    const isOverflow = (b.state as any).overflow === true;
    if (isOverflow) {
      reward -= config.overflowPenalty;
    } else {
      if (b.shotFromFarZone === true) {
        reward += config.scoreFarWeight;
      } else {
        reward += config.scoreCloseWeight;
      }
    }
  }

  // 2. Intake success reward: increase in hopper size
  const hopperDiff = r.hopper.length - prevR.hopper.length;
  if (hopperDiff > 0) {
    reward += hopperDiff * config.intakeWeight;
  }

  // 3. Motif Match reward: extra incentive for picking up target colors
  if (hopperDiff > 0) {
    const newBallColor = r.hopper[r.hopper.length - 1];
    if (world.motif.includes(newBallColor)) {
      reward += config.motifMatchWeight;
    }
  }

  // 4. Foul Penalties
  const fouls = world.match.fouls[r.alliance];
  const prevFouls = prevWorld.match.fouls[r.alliance];
  const minorDiff = fouls.minor - prevFouls.minor;
  const majorDiff = fouls.major - prevFouls.major;
  if (minorDiff > 0 || majorDiff > 0) {
    reward -= (minorDiff * 1.0 + majorDiff * 3.0) * config.foulWeight;
  }

  // 4b. Gate timing reward/penalty: fires once on the exact step the gate transitions closed→open.
  // < 7 balls: scaled penalty (more short = bigger penalty)
  // > 7 balls: scaled bonus (more surplus = bigger bonus)
  const gateJustOpened = !prevWorld.goals[r.alliance].gateOpen && world.goals[r.alliance].gateOpen;
  if (gateJustOpened) {
    const railCount = world.balls.filter(
      (b) => b.state.kind === 'rail' && b.state.goal === r.alliance && !b.state.overflow
    ).length;
    if (railCount < 7) {
      const shortfall = 7 - railCount;
      reward -= shortfall * config.earlyGatePenalty;
    } else if (railCount > 7) {
      const surplus = railCount - 7;
      reward += surplus * config.fullGateBonus;
    }
  }

  // 5. Constant step time penalty
  reward -= config.timePenalty;

  // 6. Navigation guidelines (Potential-based rewards based on distance differences)
  if (r.hopper.length === prevR.hopper.length) {
    // Ball approach: active whenever hopper isn't full
    if (r.hopper.length < 3) {
      const prevGround = prevWorld.balls.filter((b) => b.state.kind === 'ground');
      const currGround = world.balls.filter((b) => b.state.kind === 'ground');
      if (prevGround.length > 0 && currGround.length > 0) {
        let prevNearestBall = prevGround[0];
        let prevMinD = dist(prevNearestBall.pos, prevR.pos);
        for (const b of prevGround) {
          const d = dist(b.pos, prevR.pos);
          if (d < prevMinD) {
            prevMinD = d;
            prevNearestBall = b;
          }
        }
        let currNearestBall = currGround[0];
        let currMinD = dist(currNearestBall.pos, r.pos);
        for (const b of currGround) {
          const d = dist(b.pos, r.pos);
          if (d < currMinD) {
            currMinD = d;
            currNearestBall = b;
          }
        }

        const distDiff = prevMinD - currMinD;

        if (currMinD < 50.0) {
          // Nearby ball: reward approach scaled by intake alignment (smooth gradient)
          const toBallX = currNearestBall.pos.x - r.pos.x;
          const toBallY = currNearestBall.pos.y - r.pos.y;
          const toBallLen = Math.sqrt(toBallX * toBallX + toBallY * toBallY);
          if (toBallLen > 0) {
            const toBallUnit = { x: toBallX / toBallLen, y: toBallY / toBallLen };
            const alignment = Math.cos(r.heading) * toBallUnit.x + Math.sin(r.heading) * toBallUnit.y;
            const alignmentFactor = Math.max(0, alignment);
            reward += distDiff * config.ballApproachWeight * alignmentFactor;
          }
        } else {
          // Far ball: stronger pull — scale up by distance so far balls matter much more.
          // /60 instead of /144: at 120in (near field edge) gives 3x weight, strongly pulling robot across field.
          const distanceScale = 1.0 + (currMinD / 60.0);
          reward += distDiff * config.ballApproachWeight * distanceScale;
        }
      }
    }

    // Shoot zone approach: active only when the robot has 3 balls (full hopper).
    // This forces the agent to accumulate 3 balls before going to shoot.
    if (r.hopper.length === 3) {
      const closeOnly = config.scoreFarWeight === 0.0;
      const prevZoneD = distanceToLaunchZone(prevR.pos, prevR.alliance, closeOnly);
      const currZoneD = distanceToLaunchZone(r.pos, r.alliance, closeOnly);
      const distDiff = prevZoneD - currZoneD;
      reward += distDiff * config.shootZoneApproachWeight;
    }
  }

  // Gate approach penalty: active when own classifier has < 9 balls
  // Absolute per-step penalty, scaling linearly up to gateApproachPenalty at distance=0, and reaching 0 at distance >= 36
  const ownClassifierCount = world.balls.filter(
    (b) => b.state.kind === 'rail' && b.state.goal === r.alliance && !b.state.overflow
  ).length;

  if (ownClassifierCount < 9 && config.gateApproachPenalty > 0) {
    const gateRect = gateArmRect(r.alliance);
    const gateCenterPos = { x: (gateRect.x0 + gateRect.x1) / 2, y: (gateRect.y0 + gateRect.y1) / 2 };
    const currGateD = dist(r.pos, gateCenterPos);
    if (currGateD < 36.0) {
      const proximity = 1.0 - currGateD / 36.0;
      reward -= proximity * config.gateApproachPenalty;
    }
  }

  // 7. Intake Alignment Reward (potential-based, only for nearby balls)
  const currGround = world.balls.filter((b) => b.state.kind === 'ground');
  const prevGround = prevWorld.balls.filter((b) => b.state.kind === 'ground');
  if (r.hopper.length < 3 && currGround.length > 0 && prevGround.length > 0) {
    let nearestBall = currGround[0];
    let minD = dist(nearestBall.pos, r.pos);
    for (const b of currGround) {
      const d = dist(b.pos, r.pos);
      if (d < minD) {
        minD = d;
        nearestBall = b;
      }
    }

    if (minD < 50.0) {
      const toBallX = nearestBall.pos.x - r.pos.x;
      const toBallY = nearestBall.pos.y - r.pos.y;
      const toBallLen = Math.sqrt(toBallX * toBallX + toBallY * toBallY);
      
      let prevNearestBall = prevGround[0];
      let prevMinD = dist(prevNearestBall.pos, prevR.pos);
      for (const b of prevGround) {
        const d = dist(b.pos, prevR.pos);
        if (d < prevMinD) {
          prevMinD = d;
          prevNearestBall = b;
        }
      }
      const prevToBallX = prevNearestBall.pos.x - prevR.pos.x;
      const prevToBallY = prevNearestBall.pos.y - prevR.pos.y;
      const prevToBallLen = Math.sqrt(prevToBallX * prevToBallX + prevToBallY * prevToBallY);

      if (toBallLen > 0 && prevToBallLen > 0) {
        const toBallUnit = { x: toBallX / toBallLen, y: toBallY / toBallLen };
        const prevToBallUnit = { x: prevToBallX / prevToBallLen, y: prevToBallY / prevToBallLen };

        // Dot product of robot heading unit vector and target direction unit vector
        const currentAlignment = Math.cos(r.heading) * toBallUnit.x + Math.sin(r.heading) * toBallUnit.y;
        const prevAlignment = Math.cos(prevR.heading) * prevToBallUnit.x + Math.sin(prevR.heading) * prevToBallUnit.y;

        const alignDiff = currentAlignment - prevAlignment;
        const proximityFactor = 1.0 - (minD / 50.0);
        reward += alignDiff * config.intakeAlignmentWeight * proximityFactor;
      }
    }
  }

  // 8. Absolute Intake Alignment Reward: steady reward for currently facing nearby balls
  // Unlike section 7's differential signal, this fires every step the robot is well-aligned,
  // providing a stable gradient even when angle change is very small.
  if (r.hopper.length < 3) {
    const absGround = world.balls.filter((b) => b.state.kind === 'ground');
    if (absGround.length > 0) {
      let absNearestBall = absGround[0];
      let absMinD = dist(absNearestBall.pos, r.pos);
      for (const b of absGround) {
        const d = dist(b.pos, r.pos);
        if (d < absMinD) {
          absMinD = d;
          absNearestBall = b;
        }
      }

      if (absMinD < 60.0) {
        const dx = absNearestBall.pos.x - r.pos.x;
        const dy = absNearestBall.pos.y - r.pos.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len > 0) {
          // alignment in [-1, 1]: 1 = perfectly facing, -1 = facing away
          const alignment = (Math.cos(r.heading) * dx + Math.sin(r.heading) * dy) / len;
          const posAlignment = Math.max(0, alignment); // only reward facing towards
          // Scale by proximity: closer balls give stronger signal
          const proximityFactor = 1.0 - (absMinD / 60.0);
          reward += posAlignment * proximityFactor * config.intakeAlignmentAbsWeight;
        }
      }
    }
  }

  // 9. Auto Opposite Side Penalty: running penalty per step when crossing centerline during auto
  if (world.match.phase === 'auto' && config.autoOppositeSidePenalty > 0) {
    const g = goalSide(r.alliance);
    if (r.pos.x * g < 0) {
      reward -= config.autoOppositeSidePenalty;
    }
  }

  return reward;
}
