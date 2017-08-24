def run_sol(flags):
    if flags.sol == "43":
        from comps.instacart.sol43.run import run
    print("run competition instacart solution %s"%flags.sol)
    run(flags)

