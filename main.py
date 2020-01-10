import subprocess

def runEx1():    
    subprocess.call(['python', 'ex1/app.py'], shell=True) 

def runEx2():  
    subprocess.call(['python', 'ex2/app.py'], shell=True) 

def switch_example(arg):
    switcher = {
        1: runEx1,
        2: runEx2
    }

    func = switcher.get(arg, lambda: print("Invalid Example"))
    return func()

while True:
    ex = input("Run Example: ")
    if ex == '':
        continue
    if ex == "q" or ex == "Q":
        break
    switch_example(int(ex))
