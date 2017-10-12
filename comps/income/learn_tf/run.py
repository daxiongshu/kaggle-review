import os

def run(flags):

    if flags.task == "poke":
        from comps.income.income_db import incomeDB
        myDB = incomeDB(flags)
        myDB.poke()
    elif flags.task == "cv":
        from comps.income.learn_tf.cv import cv
        cv(flags,2000)        
    else:
        print("unknown task",flags.task)
        assert 0
