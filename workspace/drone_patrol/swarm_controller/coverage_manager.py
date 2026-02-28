"""
CoverageManager class for intelligent orchard patrol coverage tracking.

This module handles:
- Grid-based coverage tracking
- Visit counting and timestamping
- Least-recently-visited cell detection
- Coverage metrics (percentage, overlap)
"""

import numpy as np


class CoverageManager:
    """
    Manages coverage tracking for orchard patrol using a 2D grid.
    """
    
    def __init__(self, xmin=0, xmax=50, ymin=0, ymax=50, grid_resolution=1.0):
        """
        Initialize coverage manager with grid-based tracking.
        
        Args:
            xmin, xmax: X-axis boundaries (meters)
            ymin, ymax: Y-axis boundaries (meters)
            grid_resolution: Size of each grid cell (meters)
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.resolution = grid_resolution
        
        # Calculate grid dimensions
        self.width = int((xmax - xmin) / grid_resolution)
        self.height = int((ymax - ymin) / grid_resolution)
        
        # Create tracking grids
        self.visit_count = np.zeros((self.height, self.width), dtype=int)
        self.last_visit_time = np.zeros((self.height, self.width), dtype=float)
        
    def _world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.
        
        Args:
            x, y: World coordinates (meters)
            
        Returns:
            (row, col) grid indices, or None if out of bounds
        """
        if x < self.xmin or x >= self.xmax or y < self.ymin or y >= self.ymax:
            return None
            
        col = int((x - self.xmin) / self.resolution)
        row = int((y - self.ymin) / self.resolution)
        
        # Clamp to valid range
        col = max(0, min(col, self.width - 1))
        row = max(0, min(row, self.height - 1))
        
        return (row, col)
    
    def _grid_to_world(self, row, col):
        """
        Convert grid indices to world coordinates (cell center).
        
        Args:
            row, col: Grid indices
            
        Returns:
            (x, y) world coordinates (meters)
        """
        x = self.xmin + (col + 0.5) * self.resolution
        y = self.ymin + (row + 0.5) * self.resolution
        return (x, y)
    
    def update_coverage(self, drone_position, current_time):
        """
        Update coverage tracking for a drone's current position.
        
        Args:
            drone_position: Array [x, y, z] or [x, y]
            current_time: Current simulation time (seconds)
        """
        x, y = drone_position[0], drone_position[1]
        grid_pos = self._world_to_grid(x, y)
        
        if grid_pos is not None:
            row, col = grid_pos
            self.visit_count[row, col] += 1
            self.last_visit_time[row, col] = current_time
    
    def get_coverage_percentage(self):
        """
        Calculate percentage of orchard that has been visited.
        
        Returns:
            Coverage percentage (0-100)
        """
        visited_cells = np.sum(self.visit_count > 0)
        total_cells = self.width * self.height
        
        if total_cells == 0:
            return 0.0
            
        return (visited_cells / total_cells) * 100.0
    
    def get_overlap_ratio(self):
        """
        Calculate percentage of cells visited more than twice.
        
        Returns:
            Overlap percentage (0-100)
        """
        visited_cells = np.sum(self.visit_count > 0)
        
        if visited_cells == 0:
            return 0.0
        
        # Count cells with excessive visits (> 2)
        overlap_cells = np.sum(self.visit_count > 2)
        
        return (overlap_cells / visited_cells) * 100.0
    
    def get_frontier_cells(self):
        """
        Find frontier cells - visited cells adjacent to unvisited cells.
        
        A frontier cell is:
        - visit_count > 0 (visited)
        - Has at least one 4-connected neighbor with visit_count == 0 (unvisited)
        
        Returns:
            List of (row, col) grid indices for frontier cells
        """
        frontier = []
        
        for row in range(self.height):
            for col in range(self.width):
                # Skip unvisited cells
                if self.visit_count[row, col] == 0:
                    continue
                
                # Check 4-connected neighbors
                neighbors = [
                    (row - 1, col),  # North
                    (row + 1, col),  # South
                    (row, col - 1),  # West
                    (row, col + 1),  # East
                ]
                
                for n_row, n_col in neighbors:
                    # Check bounds
                    if 0 <= n_row < self.height and 0 <= n_col < self.width:
                        if self.visit_count[n_row, n_col] == 0:
                            frontier.append((row, col))
                            break  # Found at least one unvisited neighbor
        
        return frontier
    
    def get_least_recently_visited_cell(self, xmin, xmax):
        """
        Find the least recently visited cell within a sector.
        
        Prioritizes:
        1. Unvisited cells (visit_count == 0)
        2. Least recently visited cells (smallest last_visit_time)
        
        Args:
            xmin, xmax: Sector boundaries in x-axis (meters)
            
        Returns:
            (x, y) world coordinates of target cell
        """
        # Convert sector bounds to grid indices
        col_min = max(0, int((xmin - self.xmin) / self.resolution))
        col_max = min(self.width, int((xmax - self.xmin) / self.resolution))
        
        if col_min >= col_max:
            # Fallback: return sector center
            return ((xmin + xmax) / 2, (self.ymin + self.ymax) / 2)
        
        # Extract sector slice
        sector_visit_count = self.visit_count[:, col_min:col_max]
        sector_visit_time = self.last_visit_time[:, col_min:col_max]
        
        # Find unvisited cells first
        unvisited_mask = (sector_visit_count == 0)
        
        if np.any(unvisited_mask):
            # Choose random unvisited cell to avoid all drones going to same spot
            unvisited_indices = np.argwhere(unvisited_mask)
            if len(unvisited_indices) > 0:
                idx = np.random.choice(len(unvisited_indices))
                local_row, local_col = unvisited_indices[idx]
                global_col = col_min + local_col
                return self._grid_to_world(local_row, global_col)
        
        # All cells visited - find least recently visited
        min_time_idx = np.unravel_index(
            np.argmin(sector_visit_time), 
            sector_visit_time.shape
        )
        local_row, local_col = min_time_idx
        global_col = col_min + local_col
        
        return self._grid_to_world(local_row, global_col)
    
    def get_random_unvisited_cell(self, xmin, xmax):
        """
        Return random unvisited cell in sector where visit_count == 0.
        
        Args:
            xmin, xmax: Sector boundaries in x-axis (meters)
            
        Returns:
            (x, y) world coordinates of random unvisited cell,
            or None if all cells in sector are visited
        """
        col_min = max(0, int((xmin - self.xmin) / self.resolution))
        col_max = min(self.width, int((xmax - self.xmin) / self.resolution))
        
        if col_min >= col_max:
            return None
        
        sector_visit_count = self.visit_count[:, col_min:col_max]
        unvisited_mask = (sector_visit_count == 0)
        
        if np.any(unvisited_mask):
            unvisited_indices = np.argwhere(unvisited_mask)
            idx = np.random.choice(len(unvisited_indices))
            local_row, local_col = unvisited_indices[idx]
            global_col = col_min + local_col
            return self._grid_to_world(local_row, global_col)
        
        return None
    
    def get_best_frontier_target(self, drone_position, xmin, xmax):
        """
        Find the best frontier cell to explore within a sector.
        
        PHASE 4.2 - FIX LOCAL GREEDY COLLAPSE:
        Selects FARTHEST frontier instead of closest to force outward expansion.
        This prevents local oscillation and ensures global coverage growth.
        
        Args:
            drone_position: Current drone position [x, y, z] or [x, y]
            xmin, xmax: Sector boundaries in x-axis (meters)
            
        Returns:
            (x, y) world coordinates of target cell
        """
        minimum_target_distance = 1.0  # Don't return targets too close to drone
        
        # Get all frontier cells
        frontier_cells = self.get_frontier_cells()
        
        if not frontier_cells:
            # No frontier - try random unvisited cell first
            random_target = self.get_random_unvisited_cell(xmin, xmax)
            if random_target:
                return random_target
            
            # Final fallback
            return self.get_least_recently_visited_cell(xmin, xmax)
        
        # Filter frontiers within sector bounds
        col_min = max(0, int((xmin - self.xmin) / self.resolution))
        col_max = min(self.width, int((xmax - self.xmin) / self.resolution))
        
        sector_frontiers = [
            (row, col) for row, col in frontier_cells
            if col_min <= col < col_max
        ]
        
        if not sector_frontiers:
            # No frontier in sector - fallback to random unvisited
            random_target = self.get_random_unvisited_cell(xmin, xmax)
            if random_target:
                return random_target
            return self.get_least_recently_visited_cell(xmin, xmax)
        
        # Find drone position
        drone_x, drone_y = drone_position[0], drone_position[1]
        
        # Compute distances to all sector frontiers
        frontier_distances = []
        for row, col in sector_frontiers:
            fx, fy = self._grid_to_world(row, col)
            distance = np.sqrt((fx - drone_x)**2 + (fy - drone_y)**2)
            
            # Only consider frontiers beyond minimum distance
            if distance > minimum_target_distance:
                frontier_distances.append((distance, fx, fy))
        
        if not frontier_distances:
            # All frontiers too close - pick random unvisited cell
            random_target = self.get_random_unvisited_cell(xmin, xmax)
            if random_target:
                return random_target
            return self.get_least_recently_visited_cell(xmin, xmax)
        
        # CRITICAL FIX: Sort by distance and take FARTHEST instead of closest
        # This forces outward exploration wave expansion
        frontier_distances.sort(key=lambda x: x[0], reverse=True)  # Descending order
        
        # Take top 5 farthest and randomly choose one
        top_frontiers = frontier_distances[:5]
        chosen = top_frontiers[np.random.randint(0, len(top_frontiers))]
        _, fx, fy = chosen
        
        return (fx, fy)
    
    def get_sector_coverage(self, xmin, xmax):
        """
        Calculate coverage percentage for a specific sector.
        
        Args:
            xmin, xmax: Sector boundaries in x-axis (meters)
            
        Returns:
            Coverage percentage for this sector (0-100)
        """
        col_min = max(0, int((xmin - self.xmin) / self.resolution))
        col_max = min(self.width, int((xmax - self.xmin) / self.resolution))
        
        if col_min >= col_max:
            return 0.0
        
        sector_visit_count = self.visit_count[:, col_min:col_max]
        visited_cells = np.sum(sector_visit_count > 0)
        total_cells = sector_visit_count.size
        
        if total_cells == 0:
            return 0.0
            
        return (visited_cells / total_cells) * 100.0
    
    def __repr__(self):
        return (f"CoverageManager(grid={self.width}x{self.height}, "
                f"resolution={self.resolution}m, "
                f"coverage={self.get_coverage_percentage():.1f}%)")

