# M20Pro Sim Scripts

These scripts are local simulation helpers. They do not SSH to 104, install systemd units, record real bags, or control the robot.

```bash
./scripts/start_sim.sh true
./scripts/status_sim.sh
./scripts/stop_sim.sh
```

Use `false` as the first argument to start without RViz:

```bash
./scripts/start_sim.sh false
```
