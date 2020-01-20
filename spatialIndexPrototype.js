const RESOLUTION = {
    x: 500, 
    y: 500
}

const MIN_MBR_SIZE = 10

const GRID_CELL_SIZE = 50

// Minimum Bounding Rectangle
class MBR {
    constructor(x1, y1, x2, y2) {
        this.x1 = x1
        this.y1 = y1
        this.x2 = x2
        this.y2 = y2
    }
}

const randRange = (from, to) => {
    return from + Math.floor(Math.random() * (to - from))
}

const createRandomMBRs = (size = 10) => {
    const result = []

    for (let i = 0; i < size; i++) {
        const x1 = randRange(0, RESOLUTION.x - MIN_MBR_SIZE)
        const y1 = randRange(0, RESOLUTION.y - MIN_MBR_SIZE)
        result.push(
            new MBR( x1, y1, randRange(x1 + MIN_MBR_SIZE, RESOLUTION.x), randRange(y1 + MIN_MBR_SIZE, RESOLUTION.y) )
        )
    }

    return result
}

const createSpatialIndex = (mbrs) => {
    const grid = []
    for (let i = 0; i < RESOLUTION.y / GRID_CELL_SIZE; i++) {
        grid.push([])
        for(let j = 0; j < RESOLUTION.x / GRID_CELL_SIZE; j++) {
            grid[i].push([])
        }
    }

    mbrs.forEach((item, idx) => {
        for(let i = Math.floor(item.y1 / GRID_CELL_SIZE); i < Math.ceil(item.y2 / GRID_CELL_SIZE); i++) {
            for(let j = Math.floor(item.x1 / GRID_CELL_SIZE); j < Math.ceil(item.x2 / GRID_CELL_SIZE); j++) {
                grid[i][j].push(idx) // Change to item instead of idx, using idx for more visual print
            }
        }
    })

    return grid
}

const printSpatialIndex = (sidx) => {
    for (let i = 0; i < RESOLUTION.y / GRID_CELL_SIZE; i++) {
        console.log(JSON.stringify(sidx[i]))
    }
}

const searchUsingSpatialIndex = () => {

}

const randomMbrs = createRandomMBRs()

printSpatialIndex(createSpatialIndex(randomMbrs))
console.log(randomMbrs)