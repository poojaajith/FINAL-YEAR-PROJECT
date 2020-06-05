from flask import Flask
from flask import render_template
from flask import request
import models as dbHandler
import sqlite3 as sql
import os
import admin_fun as a
import check_admingo as check
import second_py as sec

app = Flask(__name__)
con = sql.connect('my.db')
con.execute("DROP TABLE IF EXISTS first");
con.execute('create table if not exists first ("id" integer primary key autoincrement,"username" text not null,"score" integer not null)')
#con.execute("DROP TABLE IF EXISTS users");
con.execute('create table if not exists users ("id" integer primary key autoincrement,"username" text not null,"email" text not null, "file" text not null, "twitter" text null)')
con.close
@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method=='POST':
        username = request.form['username']
        email=request.form['email']
        twitter= request.form['twitter']
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename != '':
                photo.save(os.path.join(r'C:\PYCHARM\Project\skpro\New folder', photo.filename))
        file = photo.filename
        dbHandler.insertUser(username,email,file,twitter)
        users = dbHandler.retrieveUsers()
        return render_template('fileform2.html', users=users)
    else:
        return render_template('fileform2.html')
@app.route('/VIEW', methods=['GET'])
def VIEW():
    return render_template("rec_login.html")

@app.route('/req', methods=['POST'])
def req():
    if request.method == 'POST':
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename != '':
                photo.save(os.path.join(r'C:\PYCHARM\Project\skpro\REQ FOLDER', photo.filename))
    return render_template("addgo.html")

@app.route('/ad',methods=['POST'])
def ad():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            #p=a.main()
            return render_template("addgo.html")
@app.route('/first',methods=['GET'])
def first():
    ans=check.main()
    #users = dbHandler.retrieveFirst()
    return render_template('first_phase.html', users=ans)



@app.route('/second',methods=['GET'])
def second():
    return render_template('second_phase.html')


@app.route('/third',methods=['POST'])
def third():
    if request.method=='POST':
        score = request.form['score']
        req = request.form['req']
        x= sec.foo(score,req)
        return render_template('third_phase.html', answer=x)

if __name__ == '__main__':
    app.run()