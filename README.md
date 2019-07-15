# Pathfinding
## Pathfinding algorithms implemented in python

Pathfinding is a collection of pathfinding algorithms implemented in python 3

## Installation

clone the reopo, install packages and run testfile
```bash
git clone https://github.com/LeoHeller/Pathfinding
cd Pathfinding
pip install -r requirements.txt
python3.6 AlgoTest.py
```


## Maze generation
The depth-first search algorithm of maze generation is frequently implemented using backtracking:

-Make the initial cell the current cell and mark it as visited

-While there are unvisited cells

-If the current cell has any neighbours which have not been visited
- Choose randomly one of the unvisited neighbours
- Push the current cell to the stack
- Remove the wall between the current cell and the chosen cell
- Make the chosen cell the current cell and mark it as visited

-Else if stack is not empty
- Pop a cell from the stack
- Make it the current cell

## Algorithms

### A* (A-star)

![](maze-a-star.gif)

1.Initialize the open list

2.Initialize the closed list
  put the starting node on the open 
  list (you can leave its f-cost at zero)

3.While the open list is not empty

   - find the node with the least f on 
       the open list, call it "q"

   - pop q off the open list
  
   - generate q's 8 successors and set their 
       parent to q
   
   - for each successor

        - if the successor is the goal, stop search
        
        - else:
        ```python
        successor.g = q.g + distance between 
        successor and q
        successor.h = distance from goal to the successor
        ```
        (This can be done using many 
        ways, we will discuss three heuristics- 
        Manhattan, Diagonal, and Euclidean 
        Heuristics)

        ```python
          successor.f = successor.g + successor.h
        ```

        - if a node with the same position as 
            the successor is in the `OPEN` list which has a 
           lower f than a successor, skip this successor

        - if a node with the same position as 
            the successor  is in the `CLOSED` list which has
            a lower f than a successor, skip this successor
            otherwise, add  the node to the `OPEN` list
     end (for loop)
  
    - push q on the closed list
    end (while loop) 

### Pseudo-code
```
// A* finds a path from start to goal.
// h is the heuristic function. h(n) estimates the cost to reach goal from node n.
function A_Star(start, goal, h)

    // The set of discovered nodes that need to be (re-)expanded.
    // Initially, only the start node is known.
    openSet := {start}

    // For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom := an empty map

    // For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore := map with default value of Infinity
    gScore[start] := 0

    // For node n, fScore[n] := gScore[n] + h(n).
    fScore := map with default value of Infinity
    fScore[start] := h(start)

    while openSet is not empty
        current := the node in openSet having the lowest fScore[] value
        if current = goal
            return reconstruct_path(cameFrom, current)

        openSet.Remove(current)

        for each neighbor of current

            // d(current,neighbor) is the weight of the edge from current to neighbor
            // tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore := gScore[current] + d(current, neighbor)

            if tentative_gScore < gScore[neighbor]
            // This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] := current
                gScore[neighbor] := tentative_gScore
                fScore[neighbor] := gScore[neighbor] + h(neighbor)
                if neighbor not in openSet
                    openSet.Add(neighbor)
    return failure

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)