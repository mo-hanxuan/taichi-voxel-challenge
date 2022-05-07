import taichi as ti
ti.init(arch=ti.cuda, dynamic_index=True)

blockSize = 2
fieldSize = 4
graph = ti.Vector.field(4, ti.i32, shape=(fieldSize, fieldSize))
visited = ti.field(ti.i32, shape=(fieldSize, fieldSize))


@ti.kernel
def traverseMaze():
    for i, j in ti.ndrange(fieldSize, fieldSize):
        neighbors = ti.Matrix([
            [i + 1, j], [i, j + 1], [i - 1, j], [i, j - 1]
        ])
        readed = ti.Vector([0, 0, 0, 0])
        while not all(readed):
            idx = ti.random(ti.i32) % 4
            readed[idx] = 1
            k, l = neighbors[idx, 0], neighbors[idx, 1]
            if (0 <= k < fieldSize) and (0 <= l < fieldSize):
                if not ti.atomic_or(visited[k, l], 1):  # if not visited, visite now and make visited to be true in an atomic operation
                    graph[i, j][idx] = 1

@ti.kernel
def printMaze():
    for i, j in ti.ndrange(fieldSize, fieldSize):
        neighbors = ti.Matrix([
            [i + 1, j], [i, j + 1], [i - 1, j], [i, j - 1]
        ])
        for k in ti.static(range(graph[i, j].n)):
            if graph[i, j][k]:
                print("{}, {} conected_to {}, {}".format(i, j, neighbors[k, 0], neighbors[k, 1]))


traverseMaze()
printMaze()