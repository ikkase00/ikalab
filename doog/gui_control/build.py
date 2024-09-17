import tkinter as tk
import subprocess
import os


def clone_repo():
    url = 'https://github.com/ikkase00/ikalab.git'
    target = os.path.join(os.getcwd(), 'ikalab')

    try:
        desc.config(text = 'cloning into ikalab...can take a while depending on the size of weights')
        root.update()
        subprocess.run(['git', 'clone', url, target], check = True)
        desc.config(text = 'ok!')
        root.update()
    except Exception as _:
        desc.config(text = 'something went wrong...installation aborted')
        root.update()



def the_button():
    desc.config(text = 'in progress...')
    clone_repo()

root = tk.Tk()
root.title('the mysterious installer')

desc = tk.Label(root, text = 'for ikalab')
desc.pack()

clicker = tk.Button(root, text = 'click to execute', command = the_button)
clicker.pack()

root.mainloop()

