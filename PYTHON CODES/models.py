import sqlite3 as sql

def insertUser(username,email,file,twitter):
    con = sql.connect("my.db")
    cur = con.cursor()
    cur.execute("INSERT INTO users (username,email,file,twitter) VALUES (?,?,?,?)", (username,email,file,twitter))
    con.commit()
    con.close()

def retrieveUsers():
	con = sql.connect("my.db")
	cur = con.cursor()
	#X='POOJA JITH'
	cur.execute("SELECT username,email, twitter FROM users")
	users = cur.fetchall()
	con.close()
	#print(users)
	return users


def dele():
	con = sql.connect("my.db")
	cur = con.cursor()
	cur.execute("delete from first")
	con.close()



def insertFirst(username,email,twitter,score):
	con = sql.connect("my.db")
	cur = con.cursor()
	#cur.execute("delete from first")
#db.close();
	cur.execute("INSERT INTO first (username,email,twitter,score) VALUES (?,?,?,?)", (username,email,twitter,score))
	con.commit()
	con.close()


def retrieveFirst():
	con = sql.connect("my.db")
	cur = con.cursor()
	#X='POOJA JITH'
	cur.execute("SELECT username,email,score FROM first")
	users = cur.fetchall()
	con.close()
	#print(users)
	return users

def insertthird(username,email,score):
	con = sql.connect("my.db")
	cur = con.cursor()
	#cur.execute("delete from first")
#db.close();
	cur.execute("INSERT INTO third(username,email,score) VALUES (?,?,?)", (username,email,score))
	con.commit()
	con.close()



#if __name__ == "__main__":
#    retrieveUsers()
