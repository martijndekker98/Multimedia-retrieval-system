from tkinter import *
from PIL import Image
from PIL import ImageTk
import vedo
from distanceFunctions import findMostSimilar
import step4
from collections import defaultdict
import sys
from ann import query_kdtree

def get_options_to_display():

    names_file = open('db_names_file.txt', 'r')
    nr_of_names = sum(1 for line in open('db_names_file.txt'))
    for item in names_file.readlines()[1:]:
        options_list.append(item[:-2])

def window_1():

    scrollbar_tree()
    window = Tk()
    scrollbar_query(window)
    if not destroy:
        similar_dict = match(path)
        after_query(similar_dict)

def set_tree(window, listbox, value):

    global tree
    global kr
    try:
        index = listbox.curselection()[0]
    except:
        print("Please select a distance function")
        return
    kr = value
    tree = index
    listbox.destroy()
    window.destroy()

def inspect(listbox):

    global inspected

    index = listbox.curselection()[0]
    path = options_list[index]
    mesh = vedo.load(path)
    plot = mesh.show(axes=8)
    plot.close()

def selected_item(window, listbox):

    global path 
    index = listbox.curselection()[0]
    path = options_list[index]
    window.destroy()
        

def exit_current(window):

    global destroy
    window.destroy()
    destroy = True

def scrollbar_tree():

    window = Tk()
    # global repeat
    # repeat = False

    # kr=StringVar()

    L1 = Label(window, text="K/R value")
    L1.pack( side = RIGHT)
    E1 = Entry(window, bd =4)
    E1.insert(END, '5')
    E1.pack(side = RIGHT)

    # print(kr.get())

    listbox = Listbox(window, width=25)
    listbox.pack(side= LEFT, fill=BOTH)
    scrollbar = Scrollbar(window)
    scrollbar.pack(side = RIGHT, fill = BOTH)

    listbox.insert(END, 'Custom distance function')
    listbox.insert(END, 'KDTree with KNN')
    listbox.insert(END, 'KDTree with RNN')

    listbox.config(yscrollcommand = scrollbar.set)
    scrollbar.config(command = listbox.yview)

    back_btn = Button(window, text='Exit', command=lambda: window.destroy())
    # inspect_btn = Button(window, text='Inspect', command=lambda: inspect(listbox))
    btn = Button(window, text='Go', command=lambda: set_tree(window, listbox, E1.get()))
    
    # label = Label(text = 'Choose a shape to quiry')
    btn.pack(side='bottom')
    back_btn.pack(side='bottom')
    # inspect_btn.pack(side='bottom')
    listbox.pack()
    
    listbox.selection_set( first = 0 )

    window.mainloop()


def scrollbar_query(window):

    global repeat
    repeat = False

    listbox = Listbox(window, width=25)
    listbox.pack(side= LEFT, fill=BOTH)
    scrollbar = Scrollbar(window)
    scrollbar.pack(side = RIGHT, fill = BOTH)

    for item in range(len(options_list)):
        listbox.insert(END, options_list[item])

    listbox.config(yscrollcommand = scrollbar.set)
    scrollbar.config(command = listbox.yview)

    back_btn = Button(window, text='Exit', command=lambda: exit_current(window))
    inspect_btn = Button(window, text='Inspect', command=lambda: inspect(listbox))
    btn = Button(window, text='Go', command=lambda: selected_item(window, listbox))
    
    label = Label(text = 'Choose a shape to quiry')
    btn.pack(side='bottom')
    back_btn.pack(side='bottom')
    inspect_btn.pack(side='bottom')
    listbox.pack()

    window.mainloop()

def match(path):

    path = path.split('/')
    path_file = path[-1]
    path_folder = path[-2]
    similar_dict = defaultdict(int)
    
    dataframe = step4.readCSVAsDataFrame('./featuresNew.csv')
    # dataframe = step4.readCSVAsDataFrame('./DB_fixed/features.csv')
    if tree == 0:
        similar_shapes = findMostSimilar(dataframe, path_file, 5)
    else:
        similar_shapes = query_kdtree(dataframe, path_file, tree, kr)


    for i in range(len(similar_shapes)):
        name = similar_shapes[i][0]
        resultaat = dataframe.loc[dataframe['File_name'] == name]
        folder = resultaat['Subfolder'].tolist()[0]
        mesh = f'./DB_fixed/{folder}/{name}'
        dis = similar_shapes[i][1]
        screenshot = f"./screenshots/{folder}/{name}.png"
        similar_dict[i] = mesh, screenshot, dis

    similar_dict['query'] = f'./screenshots/{path_folder}/{path_file}.png'

    return similar_dict
    

    
def exit_program(window):

    global destroy
    window.destroy()
    destroy = True

def back(window):

    window.destroy()
    window_1()

def after_query(similar_dict):

    window = Tk()

    width, height = window.winfo_screenwidth(), window.winfo_screenheight()
    window.geometry('%dx%d+0+0' % (width,height))

    # frame = Frame(master=window, width=1425, height=750)
    frame = Frame(master=window, width=width, height=height)

    frame.pack()

    min_w = 10
    max_w = width-10
    portion_w = (max_w - min_w) / 5
    half_portion = (portion_w/2) - 100
    query_pic = similar_dict['query']
    if len(similar_dict) > 1:
        
        file_1 = similar_dict[0][0]
        file_1_pic = similar_dict[0][1]
        dis_1 = similar_dict[0][2]

        img_1 = Image.open(file_1_pic)
        img_1 = img_1.resize((200,200))
        photoimage_1 = ImageTk.PhotoImage(img_1)
        button_1 = Button(
            text="Shape 1",
            width=200,
            height=200,
            bg="white",
            fg="black",
            image=photoimage_1,
            command = lambda: button_click(file_1)
        )

        button_1_label = Label(text=f"Number 1 \n {file_1} \n distance = {dis_1}",
                        width=25,
                        height=4)
                        
        button_1.place(x=10+half_portion, y=300)
        button_1_label.place(x=half_portion, y = 525)

    #
    if len(similar_dict) > 2:
        file_2 = similar_dict[1][0]
        file_2_pic = similar_dict[1][1]
        dis_2 = similar_dict[1][2]

        img_2 = Image.open(file_2_pic)
        img_2 = img_2.resize((200,200))
        photoimage_2 = ImageTk.PhotoImage(img_2)
        button_2 = Button(
            text="Shape 1",
            width=200,
            height=200,
            bg="white",
            fg="black",
            image=photoimage_2,
            command = lambda: button_click(file_2)
        )        

        button_2_label = Label(text=f"Number 2 \n {file_2} \n distance = {dis_2}",
                        width=25,
                        height=5)

        button_2.place(x=10+(portion_w*1)+half_portion, y=300)
        button_2_label.place(x=(portion_w*1)+half_portion, y = 525)

    #
    if len(similar_dict) > 3:
        file_3 = similar_dict[2][0]
        file_3_pic = similar_dict[2][1]
        dis_3 = similar_dict[2][2]

        img_3 = Image.open(file_3_pic)
        img_3 = img_3.resize((200,200))
        photoimage_3 = ImageTk.PhotoImage(img_3)
        button_3 = Button(
            text="Shape 1",
            width=200,
            height=200,
            bg="white",
            fg="black",
            image=photoimage_3,
            command = lambda: button_click(file_3)
        )

        button_3_label = Label(text=f"Number 3 \n {file_3} \n distance = {dis_3}",
                    width=25,
                    height=5)

        button_3.place(x=10+(portion_w*2)+half_portion, y=300)
        button_3_label.place(x=(portion_w*2)+half_portion, y = 525)
    #
    if len(similar_dict) > 4:
        file_4 = similar_dict[3][0]
        file_4_pic = similar_dict[3][1]
        dis_4 = similar_dict[3][2]

        img_4 = Image.open(file_4_pic)
        img_4= img_4.resize((200,200))
        photoimage_4 = ImageTk.PhotoImage(img_4)
        button_4 = Button(
            text="Shape 1",
            width=200,
            height=200,
            bg="white",
            fg="black",
            image=photoimage_4,
            command = lambda: button_click(file_4)
        )

        button_4_label = Label(text=f"Number 4 \n {file_4} \n distance = {dis_4}",
                    width=25,
                    height=5)

        button_4.place(x=10+(portion_w*3)+half_portion, y=300)
        button_4_label.place(x=(portion_w*3)+half_portion, y = 525)
    
    if len(similar_dict) > 5:
        file_5 = similar_dict[4][0]
        file_5_pic = similar_dict[4][1]
        dis_5 = similar_dict[4][2]

        img_5 = Image.open(file_5_pic)
        img_5= img_5.resize((200,200))
        photoimage_5 = ImageTk.PhotoImage(img_5)
        button_5 = Button(
            text="Shape 5",
            width=200,
            height=200,
            bg="white",
            fg="black",
            image=photoimage_5,
            command = lambda: button_click(file_5)
        )

        button_5_label = Label(text=f"Number 5 \n {file_5} \n distance = {dis_5}",
                            width=25,
                            height=5)
        button_5.place(x=10+(portion_w*4)+half_portion, y=300)
        button_5_label.place(x=(portion_w*4)+half_portion, y = 525)



    def button_click(file):

        mesh_ = vedo.load(file)
        plot = mesh_.show(new=True, axes=8)
        plot.close()        

    back_button = Button(
        text="Back",
        width=10,
        height=2,
        bg="white",
        fg="black",
        command = lambda: back(window)
    )
    exit_button = Button(
        text="Exit",
        width=10,
        height=2,
        bg="white",
        fg="black",
        command = lambda: exit_program(window)
    )

    img_query = Image.open(query_pic)
    img_query= img_query.resize((200,200))
    photoimage_query = ImageTk.PhotoImage(img_query)
    Query_image = Label(
                        width=200,
                        height=200,
                        image=photoimage_query)

    Query_label = Label(text=f"Query shape \n {query_pic[14:-4]}",
                        width=20,
                        height=4)


    Query_image.place(x=(width/2)-100, y=10)
    Query_label.place(x=(width/2)-100, y=220)

    back_button.place(x=((width*3)/10)-12.5, y = height*0.85)
    exit_button.place(x= ((width*6)/10)-12.5, y = height*0.85)

    window.mainloop()

def main():
    global options_list, path, destroy
    options_list = []
    get_options_to_display()
    destroy = False
    window_1() 

main()