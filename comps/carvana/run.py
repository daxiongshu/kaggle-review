
def run_sol(flags):
    if flags.sol == "btb1":
        from comps.carvana.btb1.run import run
    elif flags.sol == "carl":
        from comps.carvana.carl.run import run
    run(flags)

