def run_sol(flags):
    if flags.sol == "43":
        from comps.instacart.sol43.run import run
    run(flags)

