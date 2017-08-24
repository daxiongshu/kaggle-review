
def run_sol(flags):
    if flags.sol == "btb1":
        from comps.carvana.btb1.run import run

    print("run competition carvana solution %s"%(flags.sol))
    run(flags)

