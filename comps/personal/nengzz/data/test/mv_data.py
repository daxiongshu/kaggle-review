import os
with open('real_stage2_ids.txt') as f:
    for line in f:
        name = 'stage2_test_text_%s.txt'%line.strip()
        os.system('mv test/%s test2/'%name)

