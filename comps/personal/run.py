def run_sol(flags):
    if flags.sol == "baobao":
        from comps.personal.baobao.run import run
    if flags.sol == "nengzz":
        from comps.personal.nengzz.run import run
    if flags.sol == "unseen":
        from comps.personal.unseen.run import run
    if flags.sol == "seen":
        from comps.personal.seen.run import run
    run(flags)

