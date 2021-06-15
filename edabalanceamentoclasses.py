import matplotlib.pyplot as plt
total_soil = 4448
total_tree = 2224


total = total_soil + total_tree
p_soil = int(round(total_soil/total,2)*100)

p_tree =  int(round(total_tree/total,2)*100)



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

 # https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html

labels = 'Imagens de Solo', 'Imagens de Árvores'
sizes = [p_soil, p_tree]
explode = (0, 0.1)  

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Balanceamento das Classes')

plt.show()
