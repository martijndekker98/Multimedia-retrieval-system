# import trimesh
import numpy as np
import vedo

#mesh = pymesh.load_mesh("benchmark/db/0/m0/m0.off")
#f = open("benchmark/db/0/m0/test.rtf", "r")
#print(f.read())
#print("READ")

#mesh = trimesh.load("benchmark/db/0/m0/m0.off")

#trimesh.points.PointCloud(np.random.rand(10, 3)).show()
#trimesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=(0, 255, 0))
#mesh.visual.color.ColorVisuals(mesh=mesh, vertex_colors=(0,255,0))
#mesh.visual.vertex_colors = trimesh.visual.random_color()
#mesh.visual.face_colors = (0, 255, 0, 255)
#trimesh.visual.objects.create_visual(face_colors=(0,255,0,255), vertex_colors=(255,0,0,255), mesh=mesh)
#mesh.visual.edge_with = 2000
#mesh.show()

c = vedo.Cone()
c.show(axes=1)