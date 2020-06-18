from tkinter import *
import chat

def fetch():
    user_input = textfield.get()

    input_seq = chat.process_input(user_input, chat.max_len, chat.tokenizer_filepath)

    bot_response = chat.getResponse(input_seq)
    messages.insert(END, "You: " + user_input)
    messages.insert(END, "Bot: " + bot_response)
    textfield.delete(0, END)

root = Tk()

root.title('Tourist Helpdesk Chatbot')

root.geometry('500x550')

frame = Frame(root)

scrollbar = Scrollbar(frame)
scrollbar.pack(side=RIGHT, fill=Y)

messages = Listbox(frame, width=80, height=30)
messages.pack(side=LEFT, fill=BOTH)

frame.pack()

textfield = Entry(root, font=('Calibri Light', 13))
textfield.pack(fill=X, padx=10, pady=10)

textfield.focus()
textfield.bind('<Return>', (lambda event: fetch()))

root.mainloop()