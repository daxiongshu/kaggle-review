def run_sol(flags):
    if flags.sol == "learn_tf":
        from comps.income.learn_tf.run import run
    run(flags)

