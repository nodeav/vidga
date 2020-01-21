const RESOLUTION = {
    x: 500, 
    y: 500
}

const MIN_MBR_SIZE = 10

const GRID_CELL_SIZE = 50

let nextId = 0

// Minimum Bounding Rectangle
class MBR {
    constructor(x1, y1, x2, y2) {
        this.id = nextId++
        this.x1 = x1
        this.y1 = y1
        this.x2 = x2
        this.y2 = y2
    }

    isIntersecting(x, y) {
        return this.x1 <= x && x <= this.x2 && this.y1 <= y && y <= this.y2
    }

    toJSON() {
        return this.id
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
                grid[i][j].push(item)
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

const searchUsingSpatialIndex = (x, y, sidx) => {
    return sidx[Math.floor(y / GRID_CELL_SIZE)][Math.floor(x / GRID_CELL_SIZE)].filter(mbr => mbr.isIntersecting(x, y))
}

const mbrCount = 500000
const randomMbrs = createRandomMBRs(mbrCount)
const randomX = randRange(0, RESOLUTION.x)
const randomY = randRange(0, RESOLUTION.y)

let d1, d2, d3, d4, d5, d6

d1 = Date.now()
const mySpatialIndex = createSpatialIndex(randomMbrs)
d2 = Date.now()

console.log(`creating index time took: ${d2 - d1}ms`)

// printSpatialIndex(mySpatialIndex)
// console.log(randomMbrs)
console.log(`number of object - ${mbrCount}`)

d3 = Date.now()
const result1 = searchUsingSpatialIndex(randomX, randomY, mySpatialIndex)
d4 = Date.now()

d5 = Date.now()
const result2 = randomMbrs.filter(mbr => mbr.isIntersecting(randomX, randomY))
d6 = Date.now()

// console.log(`searching using index (x ${randX}, y ${randY}) - ${JSON.stringify(result1)}`)
console.log(`searching using index time took: ${d4 - d3}ms`)

// console.log(`searching without index (x ${randX}, y ${randY}) - ${JSON.stringify(result2)}`)
console.log(`searching without index time took: ${d6 - d5}ms`)