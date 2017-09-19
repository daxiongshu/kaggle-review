def run_sol(flags):
    if flags.sol == "carl":
        from comps.mobike.sol_carl.run import run
    run(flags)

