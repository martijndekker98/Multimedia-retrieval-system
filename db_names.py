import os

root = './DB_fixed'
names_file = open('db_names_file.txt', 'w')
categories = []
names_dict = {}
counter = 0

for category in os.listdir(root):
    if category != '.DS_Store':
        if not category.endswith('.csv'):
            categories.append(category)
            # names_file.write(category + '\n')
            new_path = root+'/'+category
            for file_nr in os.listdir(new_path):
                if not file_nr.endswith('.ply'):
                    continue
                file_path = new_path + '/' + file_nr
                names_file.writelines(file_path + ' \n')
                counter += 1

names_file.close()





