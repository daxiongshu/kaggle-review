from comps.personal.personal_db import personalDB

def preprocess(flags):
	myDB = personalDB(flags,name="full")
	myDB.poke()
