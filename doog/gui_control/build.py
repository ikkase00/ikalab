import tkinter as tk

def the_button():
    label.config(text = 'oh..')

root = tk.Tk()
root.title('the thing')

label = tk.Label(root, text = '<-')
label.pack()

but = tk.Button(root, text = 'o', command = the_button)
but.pack()

root.mainloop()

