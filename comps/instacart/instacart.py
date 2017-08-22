def run_instacart(flags):
    if flags.sol == "43":
        run_sol43(flags)

def run_sol43(flags):
    from comps.instacart.sol43.run import run
    print("run 43th solution")
    run(flags)
