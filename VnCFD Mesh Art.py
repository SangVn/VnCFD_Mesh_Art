#!/usr/bin/env python
# coding: utf-8

# Tài liệu này mang giấy phép Creative Commons Attribution (CC BY). (c) Nguyễn Ngọc Sáng, Zhukovsky 03/2020.
# 
# [@SangVn](https://github.com/SangVn) [@VnCFD](https://vncfdgroup.wordpress.com/)
# 
# *Thực hành CFD với Python!*

# # VnCFD Mesh Art
# Cùng với lưới có cấu trúc, lưới không cấu trúc được sử dụng rộng rãi trong các chương trình mô phỏng các hiện tượng vật lý như chuyển động phân tử, chất khí, chất lỏng, chất rắn; quá trình trao đổi nhiệt; các hiện tượng điện, từ trường; các quá trình phản ứng hóa học... Ngoài ra, ta có thẻ dùng nó để tạo ra những bức hình đẹp!
# 
# 
# <img src='img/anentangleda.jpg' width=800>
# <center>(Nguồn naked-science.ru)</center>
# 
# Trong khóa học này, chúng ta sẽ học cách chia lưới không cấu trúc một miền không gian 2D bất kì và biểu diễn nó đa màu sắc bằng Python. 
#     
# Keys words: Unstructured Mesh, matplotlib.tri, Triangulation, Gmsh
# 
# ## 1. matplotlib.tri
# 
# Để tìm hiểu về lưới không cấu trúc, trước hết ta sẽ làm quen với thư viện python **matplotlib.tri**. Thư viện này cho phép ta chia và biễu diễn lưới không cấu trúc, trong đó mỗi ô lưới là một tam giác.
# Về cấu trúc, mỗi lưới là một tập hợp các ô lưới hình tam giác (triangle); mỗi tam giác được tạo thành bởi ba đỉnh (point, vertex, số nhiều: vertices), có ba cạnh (edge) và có tối đa 3 tam giác 'hàng xóm' (neighbour) theo thứ tự tương ứng với ba cạnh; mỗi cạnh gồm 2 đỉnh.
# 
# Xét ví dụ cụ thể sau:

# In[1]:


# Trước hết ta load các thư viện cần thiết 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import os


# In[2]:


# Cho tọa độ tập hợp các điểm 
x = np.asarray([0, 1, 2, 3, 0.5, 1.5, 2.5, 1, 2, 1.5, 0.0, 3.0])
y = np.asarray([0, 0, 0, 0, 1.0, 1.0, 1.0, 2, 2, 3.0, 2.0, 2.0])

# Để xác định được các tam giác, ta biểu diễn các điểm này cùng với số thứ tự của chúng 
fig = plt.figure(figsize=(4,4))
plt.scatter(x, y)
for i in range(x.size): plt.annotate(str(i), (x[i], y[i]))
plt.show()


# In[3]:


# triangle = [v1, v2, v3] - chọn thứ tự các đỉnh theo chiều ngược chiều kim đồng hồ
# Dựa vào hình trên ta xác định được list các tam giác bao gồm:
triangles = [[0, 1, 4], [1, 2, 5], [2, 3, 6], [1, 5, 4], [2, 6, 5], 
             [4, 5, 7], [5, 6, 8], [5, 8, 7], [7, 8, 9], [10, 4, 7], [8, 6, 11]]

# Để tạo cấu trúc dữ liệu lưới, ta sử dụng hàm Triangulation trong matplotlib.tri
mesh = mtri.Triangulation(x, y, triangles, mask=None)

# Để xem cấu trúc của của lớp Trangulation dùng các lệnh:
print(mesh.__doc__)

# Liệt kê danh sách các hàm 
print(dir(mesh))

# Để xem đầy đủ, chi tiết cấu trúc dùng lệnh:
# print(help(mesh))


# In[4]:


# Biểu diễn dưới 
def plot(mesh, colors = ['ko-', 'b-', 'g-'], cmaps=['viridis', 'plasma', 'hsv']):
    # Giá trị một đại lượng vật lý có thể lưu tại tâm các ô lưới hoặc tại các đỉnh lưới
    # Ví dụ:
    z_vertices = np.random.rand(len(mesh.x))
    z_cells    = np.random.rand(len(mesh.triangles))

    # Ta có thể biểu diễn lưới như sau:
    fig = plt.figure(figsize=(16, 4))
    ax = fig.subplots(1, 4)

    # chỉ biểu diễn lưới
    ax[0].triplot(mesh, colors[0])

    # biểu diễn lưới và giá trị tại đỉnh 
    ax[1].triplot(mesh, colors[1])
    ax[1].tricontourf(mesh, z_vertices, cmap=cmaps[0])

    # biểu diễn lưới và giá trị tại tâm 
    ax[2].triplot(mesh, colors[2])
    ax[2].tripcolor(mesh, z_cells, cmap=cmaps[1])

    # chỉ biểu diễn giá trị (tại đỉnh/tại tâm)
    ax[3].tripcolor(mesh, z_cells, cmap=cmaps[2])

    plt.show()
    
plot(mesh) 


# Các màu và cmap khác được liệt kê trên trang matplotlib:
# 
# https://matplotlib.org/3.1.1/gallery/color/named_colors.html
# 
# https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
# 
# **Biến 'mask' trong hàm Triangulation có ý nghĩ gì?**
# <br>Mask là một tập hợp các giá trị boolean độ dài bằng số lượng các tam giác. Nếu tam giác thứ $i$ được đánh dấu, nghĩa là $mask[i]=True$, thì nó sẽ không được biễu diễn khi plot. Ví dụ:

# In[5]:


# tạo mask với tất cả các giá trị bằng False 
mask = np.zeros(len(triangles), dtype=bool)
# đánh dấu các tam giác sau 
for i in (0, 2, -2, -1): mask[i] = True
# set mask lại cho mesh 
mesh.set_mask(mask)
plot(mesh, cmaps=['cool', 'jet', 'viridis'])
# ta thấy, không giống như tam giác, đỉnh vẫn xuất hiện


# Nếu như thay vì vẽ tất cả các cạnh, ta chỉ muốn vẽ đường biên thì làm thế nào?
# <br>Như đã nói ở trên, tam giác có tối đa 3 tam giác xung quanh tương ứng với ba cạnh, hay là triangle.neighbors=[n1, n2, n3], trường hợp khuyết hàng xóm thì chỉ số của hàng xóm đó bằng -1. Ví dụ:

# In[6]:


# để liệt kê danh sách hàng xóm của tất cả các tam giác ta dùng câu lệnh: mesh.neighbors
# để xem danh sách hàng xóm của tam giác thứ i ta dùng lệnh: mesh.neighbors[i]
# tam giác thứ 0 (đầu tiên) đã bị đánh dấu nên không có hàng xóm nào, 
# tam giác thứ 1 ([1, 2, 5]) có hai hàng xóm là 4 và 3 theo thứ tự cạnh thứ 1 và 2, cạnh 0 ko có hàng xóm là cạnh biên 
mesh.neighbors[0:2]


# In[7]:


# Như vậy, nếu tam giác không bị đánh dấu thì cạnh nào không có hàng xóm là cạnh biên.
def get_border(mesh):
    boundaries = []
    mask = mesh.mask
    if mask is not None:
        for i in range(len(mask)):
            if not mask[i]:
                for j in range(3):
                    if mesh.neighbors[i,j] < 0:
                        boundaries.append((mesh.triangles[i,j], mesh.triangles[i,(j+1)%3]))
    else:
        for i in range(len(mesh.triangles)):
                for j in range(3):
                    if mesh.neighbors[i,j] < 0:
                        boundaries.append((mesh.triangles[i,j], mesh.triangles[i,(j+1)%3]))
    return np.asarray(boundaries)


# Biểu diễn đường biên 
def plot_border(mesh, colors=['r*-', 'm-', 'g-', 'y-'], cmaps=['spring', 'summer', 'autumn']):
    border = get_border(mesh)
    xb, yb = mesh.x[border].T, mesh.y[border].T
    
    z_vertices = np.random.rand(len(mesh.x))
    z_cells    = np.random.rand(len(mesh.triangles))
    
    fig = plt.figure(figsize=(16, 4))
    ax = fig.subplots(1, 4)
    ax[0].plot(xb, yb, colors[0])
    ax[1].plot(xb, yb, colors[1])
    ax[1].tricontourf(mesh, z_vertices, cmap=cmaps[0])
    ax[2].plot(xb, yb, colors[2])
    ax[2].tripcolor(mesh, z_cells, cmap=cmaps[1])
    ax[3].plot(xb, yb, colors[3])
    ax[3].tripcolor(mesh, z_cells, cmap=cmaps[2])

    plt.show()


# In[8]:


plot_border(mesh)


# # 2. Gmsh
# 
# Trường hợp có ít điểm lưới, ta có thể dựa theo cách trên để tự mình xác định tập hợp các tam giác (triangles). Tuy nhiên, nếu có một triệu điểm thì việc này là không thể. Ta có thể sử dụng Triangulation mà ko có biến triangles, nhưng khi đó có thể xuất hiện các tam giác không mong muốn nằm ngoài biên. Ví dụ: 

# In[9]:


mesh1 = mtri.Triangulation(x, y)
z_cells = np.random.rand(len(mesh.triangles))
plot(mesh1, cmaps=['winter', 'rainbow', 'prism'])


# Vậy nên, cách nhanh nhất là sử dụng các phần mềm chia lưới chuyên nghiệp. Ở đây chúng ta sẽ sử dụng **Gmsh (http://gmsh.info/)**. Đây là một phần mềm mã nguồn mở được sử dụng rộng rãi. Hãy cài đặt Gmsh và add path và xem qua cách sử trước khi thực hiện những bước tiếp theo!
# 
# ## Gmsh tutorials
# 
# Trước hết ta tìm hiểu cấu trúc file hình học đầu vào *.geo file*. Trong file .geo khai báo các biến số, đối tượng cơ bản (points, curves, surfaces), đối tượng vật lý (points, curves, surfaces).
# 
# ### t1.geo
# 
# Chia lưới hình vuông kích thước 1x1. Nội dung file t1.geo: 

# In[ ]:


# /**************
# * Gmsh tutorial 1
# **************/

# lc = 0.25;

# Point(1) = {0.0, 0.0, 0, lc};
# Point(2) = {1.0, 0.0, 0, lc};
# Point(3) = {1.0, 1.0, 0, lc};
# Point(4) = {0.0, 1.0, 0, lc};

# Line(1) = {1, 2};
# Line(2) = {3, 2};
# Line(3) = {3, 4};
# Line(4) = {4, 1};

# Curve loop(1) = {4, 1, -2, 3}; //'-2' vì chiều của Line(2) bị ngược
# Plane Surface(1) = {1};
# /***************/


# Trong đó:
# - lc: biến số độ dài đặc trưng, dùng để xác định kích thước phần tử tại các điểm, nếu lc bằng không thì gmsh sẽ tự động thiết lập.
# - Point(i) = (x, y, z, lc): điểm trên biên, i - số thứ tự, (x,y,z) - tọa độ
# - Curve: đường, đường đơn giản nhất là đường thẳng Line
# - Line(j) = {i1, i2}: j - số thứ tự, {i1, i2} - điểm bắt đầu và điểm cuối
# - Curve Loop(k) = {j1, j2, j3,...} : tập hợp các đường 
# - Surface(l) = {k1, k2...}: mặt 
# 
# Thông tin đầy đủ: http://gmsh.info/doc/texinfo/gmsh.html
# 
# Xem thêm: https://openfoamwiki.net/index.php/2D_Mesh_Tutorial_using_GMSH
# 
# Ở đây, chạy gmsh ta sẽ dùng các câu lệnh:
# <br>[http://gmsh.info/doc/texinfo/gmsh.html#Command_002dline-options]
# 
# Ví dụ: để mở file .geo, tạo lưới 2D và xuất lưới ở định dạng .dat, ta viết như sau:

# In[10]:


os.system('gmsh -2 data/t1.geo -o data/t1.msh -format dat')


# In[11]:


# Nội dung chính file t1Mesh.dat
# (----------------Nodes----------------)
# node 1 0 0
# node 2 1 0
# ...
# (++++++++++++++ E L E M E N T S ++++++++++++++)
# element 13 -tria3  3 7 11
# element 14 -tria3  8 1 10
# ...


# node - chính là các điểm lưới, còn element là các tam giác. Ta viết một hàm để đọc file mesh.dat:

# In[12]:


def read_gmsh(filename):
    nodes = []
    elements = []
    with open(filename) as f:
        for line in f:
            if line[:4] == 'node':
                ixy = line.split()
                nodes.append([float(ixy[2]), float(ixy[3])])
            elif line[:7] == 'element':
                ijk = line.split()
                elements.append([int(ijk[3])-1, int(ijk[4])-1, int(ijk[5])-1])
    return np.array(nodes), elements


# In[13]:


# đọc và biểu diễn lưới vừa nhận được
def plot_gmsh(filename, plot_func=plot, colors=['r*-', 'm-', 'g-', 'y-'], cmaps=['spring', 'summer', 'autumn']):
    points, triangles = read_gmsh(filename)
    mesh = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    plot_func(mesh)

plot_gmsh('data/t1.msh')


# ## t2.geo
# 
# Chia lưới hình vuông có lỗ ở giữa (hãy xem nội dung file). Ở đây ta khai báo thêm đường viền bên trong, thêm một Curve loop.

# In[14]:


# Curve loop(1) = {4, 1, -2, 3};
# Curve loop(2) = {8, 5, 6, 7};
# Plane Surface(1) = {1, 2};
os.system('gmsh -2 data/t2.geo -o data/t2.msh -format dat')


# In[15]:


plot_gmsh('data/t2.msh')


# ## t3.geo
# Để chia lưới và biểu diễn cả hình vuông ở giữa, khai báo thêm Physical Surface (dùng để đặt điều kiện biên):

# In[16]:


# Plane Surface(1) = {1, 2};
# Plane Surface(2) = {2};
# Physical Surface("out") = {1};
# Physical Surface("in") = {2};

os.system('gmsh -2 data/t3.geo -o data/t3.msh -format dat')


# In[ ]:


# Khi đó trong file mesh.dat xuất hiện thêm các thông số:

# ( +------------+---------Physical Groups Section----------+------------+
# (Element sets ===> 'element_group' to identify DIFFERENT MATERIALS =out)
# 1 2 3 4 5 6 7 8 9 10
# ...
# (Element sets ===> 'element_group' to identify DIFFERENT MATERIALS =in)
# 49 50 51 52 53 54 55 56 57 58 
# ...
# (Node sets ===> Used to set BOUNDARY CONDITIONS in Tochnog =out)
# 7 6 5 4 3 2 9 1 8 10 
# ...
# (Node sets ===> Used to set BOUNDARY CONDITIONS in Tochnog =in)
# 7 6 5 8 21 22 23 24 38 39 
# ...

# Trong đó Element sets out chính là lưới hình vuông có lỗ, sets in -- lưới lỗ. Ta sửa lại hàm đọc lưới để đọc thêm sets như sau:


# In[17]:


def new_read_gmsh(filename):
    nodes = []
    elements = []
    element_sets = []

    with open(filename) as f:
        while True:
            line = f.readline()
            if line == '': break

            if line[:4] == 'node':
                ixy = line.split()
                nodes.append([float(ixy[2]), float(ixy[3])])
            elif line[:7] == 'element':
                ijk = line.split()
                elements.append([int(ijk[3])-1, int(ijk[4])-1, int(ijk[5])-1])
            elif line[:13] == '(Element sets':
                sets = []
                while True:
                    line = f.readline()
                    if line == '\n':
                        element_sets.append(sets)
                        break
                    for i in line.split():
                        sets.append(int(i))
         
    return np.array(nodes), elements, element_sets


# In[18]:


# Hàm biểu diễn mới 
# Sử dụng mask để thay đổi màu sách các vùng ô lưới (sets)

def new_plot_gmsh(filename, outfile=None, fsize = (6, 4), colors=['r-', 'b-', 'g-'],                   cmaps=['viridis', 'cool', 'autumn'], pborder=[False, False, False], paxis='off'):
    
    points, triangles, sets = new_read_gmsh(filename)
    z = np.random.rand(points.shape[0])
    fig = plt.figure(figsize=fsize)
    
    for j in range(len(sets)):
        mask = np.ones(len(triangles), dtype=bool)
        for i in sets[j]: mask[i-1] = False
        mesh = mtri.Triangulation(points[:, 0], points[:, 1], triangles, mask)
        if pborder[j] is True:
            border = get_border(mesh)
            xb, yb = mesh.x[border].T, mesh.y[border].T
            plt.plot(xb, yb, colors[j])
            
        plt.tripcolor(mesh, z, cmap=cmaps[j])
    
    plt.axis(paxis)
    if outfile is not None: fig.savefig(outfile, bbox_inches='tight')
    plt.show()


# In[19]:


new_plot_gmsh('data/t3.msh')


# In[20]:


# tương tự cho t4.geo, có hai lỗ bên trong  

# Plane Surface(1) = {1, 2, 3};
# Plane Surface(2) = {2};
# Plane Surface(3) = {3};
# Physical Surface("out") = {1};
# Physical Surface("in1") = {2};
# Physical Surface("in2") = {3};
os.system('gmsh -2 data/t4.geo -o data/t4.msh -format dat')
new_plot_gmsh('data/t4.msh')


# In[21]:


# tương tự cho t5.geo, lỗ kép 

# Plane Surface(1) = {1, 2};
# Plane Surface(2) = {2, 3};
# Plane Surface(3) = {3};
# Physical Surface("out") = {1};
# Physical Surface("in1") = {2};
# Physical Surface("in2") = {3};
os.system('gmsh -2 data/t5.geo -o data/t5.msh -format dat')
new_plot_gmsh('data/t5.msh')


# # 3. Chuyển từ ảnh sang lưới
# 
# Đối với một miền không gian bất kì, đường biên là một tập hợp rất nhiều điểm, chúng có thể được lưu ở định dạng file .dat thông thường chỉ gồm tọa độ (x, y). Ta sẽ viết thêm một hàm để chuyển từ file .dat sang file .geo. 
# 
# ## 3.1 Trường hợp chỉ có một đường biên ngoài 

# In[22]:


def convert(infile, outfile, lc=0.0):
    f = open(outfile, 'w')
    f.write('lc=%8.8f;\n' % lc)

    points = np.loadtxt(infile, delimiter=',')
    Np = points.shape[0]
    for i in range(Np):
        f.write('Point(%d) = {%8.8f, %8.8f, 0.0, %8.8f};\n' % (i+1, points[i, 0], points[i, 1], lc))
        
    for i in range(Np-1):
        f.write('Line(%d) = {%d, %d};\n' % (i+1, i+1, i+2))
    f.write('Line(%d) = {%d, %d};\n' % (Np, Np, 1))
    
    f.write('Curve Loop(1) = { 1')
    for i in range(1, Np): f.write(', %d' % (i+1))
    f.write(' };\n')
    
    f.write('Plane Surface(1) = {1};\n')       
    f.close()


# In[23]:


# Ví dụ 
convert('data/Vietnam.dat', 'data/Vietnam.geo')
os.system('gmsh -2 data/Vietnam.geo -o data/Vietnam.msh -format dat')


# In[24]:


# Viết một hàm plot mới có khả năng vẽ cùng một lúc nhiều trường hợp khác nhau 
def vncfd_mesh_art(meshfile, outfile = None, fsize=(16, 12), subplots=[1, 3], pborder=False,                   colors=['k-', 'r-', 'y-'], cmaps=['viridis', 'jet', 'autumn'], paxis='on', y=0.9):
    nodes, elements = read_gmsh(meshfile)
    mesh = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    border = get_border(mesh)
    xb, yb = mesh.x[border].T, mesh.y[border].T
    
    z_vertices = np.random.rand(len(mesh.x))
    z_cells    = np.random.rand(len(mesh.triangles))

    fig = plt.figure(figsize=fsize)
    nrows, ncols = subplots[0], subplots[1]
    axs = fig.subplots(nrows, ncols)
    ax = axs.flat
    
    for i in range(nrows*ncols):
        if pborder is True: ax[i].plot(xb, yb, colors[i])
        else: ax[i].triplot(mesh, colors[i])
        
        if i==0: ax[i].tricontourf(mesh, z_vertices, cmap=cmaps[i])
        else: ax[i].tripcolor(mesh, z_vertices, cmap=cmaps[i])
        
        ax[i].axis(paxis) # lựa chọn vẽ trục tọa độ hay ẩn
    
    ax[0].set_ylim(ax[1].get_ylim())
    ax[0].set_xlim(ax[1].get_xlim())
    
    fig.suptitle('VnCFD Mesh Art', fontsize=16, fontweight='bold', y=y)
    if outfile is not None: 
        if outfile is 'auto': outfile = (meshfile.replace('data', 'img')).replace('.msh', '.png')
        fig.savefig(outfile, bbox_inches='tight')
    
    plt.show()


# In[25]:


vncfd_mesh_art('data/Vietnam.msh', outfile='auto')


# In[26]:


# Ví dụ chia lưới hình con mèo 
convert('data/Cat.dat', 'data/Cat.geo')
os.system('gmsh -2 data/Cat.geo -o data/Cat.msh -format dat')


# In[27]:


vncfd_mesh_art('data/Cat.msh', fsize=(16, 6), outfile='auto', pborder=False, y=0.95)


# **Làm thế nào để có được tọa độ hình con mèo như trên?**
# 
# Chúng ta sẽ làm việc này một cách thủ công:
# * Bước 1. nếu bạn ko biết vẽ thì tìm một hình vẽ mà bạn thích, download về (hình dễ lấy đường biên nhất là hình silhouette.
# 
# * Bước 1. sử dụng trang WebPlotDigitizer https://automeris.io/WebPlotDigitizer/. Khi bạn cần lấy dữ liệu từ một đồ thị trong một bài báo hay một quyển sách cũng có thể dùng trang này. Cách sử dụng rất đơn giản, chỉ cần upload ảnh lên trang, chọn 4 điểm [x1, x2, y1, y2] trên hai trục tọa độ [x, y], xác định chiều dài tương ứng hai đoạn, sau đó bắt đầu chọn các điểm (theo thứ tự ngược chiều kim đồng hồ), lấy dữ liệu vào lưu lại. Để đơn giản hãy chọn x1, y1 trùng nhau và là gốc tọa độ hay là điểm góc phía dưới bên trái, y2 - điểm góc trên bên trái, x2 - góc dưới bên phải, khi đó chiều dài hai đoạn chính là kích thước bức ảnh (click chuột phải chọn properties). 
# 
# Ví dụ: **thực hành với hình whale.jpg trong thư mục img:**
# 

# In[28]:


# Ví dụ chia lưới hình cá voi
convert('data/Whale.dat', 'data/Whale.geo')
os.system('gmsh -2 data/Whale.geo -o data/Whale.msh -format dat')


# In[29]:


vncfd_mesh_art('data/Whale.msh', outfile='auto', fsize=(22, 14), subplots=[2, 1], pborder=False, y=0.9)


# In[30]:


# Ví dụ chia lưới hình sói 
convert('data/Wolf.dat', 'data/Wolf.geo')
os.system('gmsh -2 data/Wolf.geo -o data/Wolf.msh -format dat')


# In[31]:


vncfd_mesh_art('data/Wolf.msh', outfile='auto', fsize=(16, 14), subplots=[2, 3], pborder=False,               colors=['k-', 'r-', 'y-', 'b-', 'g-', 'm-'],                cmaps=['viridis', 'jet', 'autumn', 'hsv', 'magma', 'cool'], paxis='on', y=0.9)


# # 3.2 Trường hợp tổng quát
# 
# Trường hợp khi có nhiều đường biên, trong file hình học .dat, ngoài tọa độ các điểm, ta cần chỉ rõ **có bao nhiêu đường biên, mỗi đường gồm bao nhiêu điểm, và viết luôn phần Plane Surface, Physical Surface vào đầu file**. Quy ước cách viết như ở ví dụ chia lưới cho profile (xem nội dung file naca_23012.dat).
# 
# **Ví dụ 1: chia lưới naca_23012**
# Number of curve loops: 2
# Number of points of each curveloop: 100 4
# Plane Surface(1) = {1, 2};
# Plane Surface(2) = {2};
# Physical Surface("out") = {1};
# Physical Surface("in") = {2};
# Curve Loop 1
9.924038765061040657e-01 -8.682408883346515172e-02
...
# Curve Loop 2
-0.5 -0.75
...
# In[32]:


# Viết lại hàm convert 
def new_convert(infile, outfile, lc=0.0):
    try: # đọc tọa độ các điểm bằng loadtxt 
        points = np.loadtxt(infile, delimiter=',')
    except:
        points = np.loadtxt(infile)
        
    Np = points.shape[0]
    
    NCurves = 1
    StartLines = [1]
    Surface = ''
    with open(infile, 'r') as f:
        # đọc số Curve Loops 
        line1 = f.readline().split()
        NCurves = int(line1[-1])
        
        # đọc số điểm trên mỗi loop, xách định chỉ số của đường đầu tiên trong loop 
        line2 = f.readline().split()
        for i in range(NCurves):
            N = int(line2[-(NCurves-i)])
            StartLines.append(StartLines[i]+N)
        # đọc phần surface
        while True:
            line = f.readline().replace('# ', '')
            if line[:5] == 'Curve': break # hết phần khai báo mặt 
            else: Surface += line
    
    f = open(outfile, 'w')
    f.write('lc=%8.8f;\n\n' % lc)

    for i in range(Np):
        f.write('Point(%d) = {%8.8f, %8.8f, 0.0, %8.8f};\n' % (i+1, points[i, 0], points[i, 1], lc))
    
    f.write('\n')
    for j in range(NCurves):
        for i in range(StartLines[j], StartLines[j+1]-1):
            f.write('Line(%d) = {%d, %d};\n' % (i, i, i+1))
        f.write('Line(%d) = {%d, %d};\n' % (StartLines[j+1]-1, StartLines[j+1]-1, StartLines[j]))

        f.write('\nCurve Loop(%d) = { %d' % (j+1, StartLines[j]))
        for i in range(StartLines[j], StartLines[j+1]-1): f.write(', %d' % (i+1))
        f.write(' };\n\n')
    
    f.write(Surface)    
    f.close()


# In[33]:


# convert, chia lưới 
new_convert('data/naca_23012.dat', 'data/naca_23012.geo', lc=0.5)
os.system('gmsh -2 data/naca_23012.geo -o data/naca_23012.msh -format dat')


# In[34]:


# biểu diễn lưới 
new_plot_gmsh('data/naca_23012.msh', 'img/naca_23012.png', fsize = (8, 6), cmaps=['winter', 'autumn'])


# **Ví dụ 2: Chia dòng chữ VnCFD**
# <img src='img/VnCFD.png' width=400>
# Các bước lấy tọa độ thủ công như hướng dẫn ở trên. Chú ý trong trường hợp này ta sẽ có 7 vùng không gian cần phối màu: vùng bao ngoài, chữ V, chữ n, chữ C, chữ F, chữ D, lõi bên trong chữ D.
# File VnCFD.dat có nội dung như sau:

# In[35]:


# Number of curve loops: 7
# Number of points of each curveloops: 19 37 48 40 22 17 4
# Plane Surface(1) = {1, 2, 3, 4, 5, 7}; //vùng bao ngoài
# Plane Surface(2) = {1};                //chữ V
# Plane Surface(3) = {2};                //chữ n
# Plane Surface(4) = {3};                //chữ C
# Plane Surface(5) = {4};                //chữ F
# Plane Surface(6) = {5,6};              //chữ D
# Plane Surface(7) = {6};                //lõi chữ D 
# Physical Surface("out") = {1};
# Physical Surface("V") = {2};
# Physical Surface("n") = {3};
# Physical Surface("C") = {4};
# Physical Surface("F") = {5};
# Physical Surface("D") = {6};
# Physical Surface("inD") = {7};
# Curve Loop 1 'V'
#....


# In[36]:


# convert và chia lưới 
new_convert('data/VnCFD.dat', 'data/VnCFD.geo', lc=50)
os.system('gmsh -2 data/VnCFD.geo -o data/VnCFD.msh -format dat')


# In[37]:


# plot và lưu hình 
new_plot_gmsh('data/VnCFD.msh', 'img/VnCFD.jpg', fsize = (24, 10.5), colors=['r-','r-','r-','r-','r-','r-','r-'],             cmaps=['Purples','jet', 'autumn', 'cool', 'hsv', 'plasma', 'Purples'],             pborder=[False, True, True, True, True, True, False])


# # Kết luận
# 
# Trên đây là các bước cơ bản để chia và biểu diễn lưới một miền không gian hai chiều bất kì. Bằng cách thay đổi các thông số colors, cmaps, mask bạn sẽ có được những hình ảnh đầy màu sắc.
# 
# Tuy nhiên, việc lấy tọa độ thủ công khá mất thời gian với những miền phức tạp, thế nên hãy tìm hiểu viết hàm tự động làm việc này sử dụng openCV, nếu bạn hứng thú với việc chia và biểu diễn lưới.

# In[ ]:




