import numpy as np
import matplotlib.pyplot as plt


class QPD:
    def __init__(self, size: float, gap: float, shape: str, n: int = 1000):
        """

        Parameters
        ----------
        size : float
            Size of the detector. Depending on the parameter "shape" this is either the length or the diameter. (in m)
        gap : float
            Width of the gap between the quadrants of the detector (in m)
        shape : str
            Defines the shape of the detector. "square": detector is square shaped,
            "circular": detector is circular shaped
            Depending on the value of "shape", the meaning of the parameter "size" is accordingly.
        n : int, optional
            Number of elements to divide detector into along each axis, i.e. the total number elements is n**2.
            This value will be rounded up to the next higher even number. Defaults to 1,000.
        """
        self._size = self.size = size
        self._gap = self.gap = gap
        self._n = self.n = n
        self._laser_intensity = self.laser_intensity = np.zeros((self._n, self._n))

        self._detector_intensity = np.zeros((self._n, self._n))
        # Parameters describing the geometry of the detector. Will be computed in the method "create_detector"
        self._detector_active = None
        self._detector_x = None
        self._detector_y = None

        # Create detector
        self.create_detector(shape)

    ## ==================================== Properties ================================================================
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @property
    def gap(self):
        return self._gap

    @gap.setter
    def gap(self, value):
        self._gap = value

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value % 2:
            self._n = value + 1  # round up to next even number
        else:
            self._n = value

    @property
    def detector_active(self):
        return self._detector_active

    @property
    def detector_x(self):
        return self._detector_x

    @property
    def detector_y(self):
        return self._detector_y

    @property
    def laser_intensity(self):
        return self._laser_intensity

    @laser_intensity.setter
    def laser_intensity(self, value: np.ndarray):
        """
        Sets the laser intensity

        Parameters
        ----------
        value : np.ndarray
            2d array of size (self.n, self.n) describing the distribution of a laser beam hitting the QPD.

        """
        if np.shape(value) == (self.n, self.n):
            self._laser_intensity = value
        else:
            raise(ValueError('Invalid shape for laser intensity'))

    @property
    def detector_intensity(self):
        return self._detector_active*self._laser_intensity

    @property
    def quadrants(self) -> list:
        """
        Computes the integrated intensity inside the four quadrant segments.

        The definition of the quadrants is as in
        https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=4400&pn=PDQ80A

        The (0,0) of the coordinate system is at the lower left edge of the QPD.

        Returns
        -------
        list
            List with four elements, corresponding to the intensity in the individual segments.
        """
        det = self.detector_intensity
        n = self.n

        q1 = np.sum(det[n//2+1:n, n//2+1:n])
        q2 = np.sum(det[0:n//2, n//2+1:n])
        q3 = np.sum(det[0:n // 2, 0:n // 2])
        q4 = np.sum(det[n//2+1:n, 0:n//2])

        return [q1, q2, q3, q4]

    @property
    def sum(self) -> float:
        """
        Computes the intensity sum over the whole detector.

        Returns
        -------
        float
            Integrated intensity
        """
        return sum(self.quadrants)

    @property
    def x_diff(self) -> float:
        """
        Computes the "x_diff", i.e. the difference between the sum of the left and right quadrants.
        The sign is choosen such that a beam on the "right" side leads to positive values.
        Returns
        -------
        float
            x_diff
        """
        q1, q2, q3, q4 = self.quadrants
        return -(q2+q3) + (q1+q4)

    @property
    def y_diff(self) -> float:
        """
        Computes the "y_diff", i.e. the difference between the sum of the top and bottom quadrants.
        The sign is choosen such that a beam on the "top" side leads to positive values.
        Returns
        -------
        float
            y_diff
        """

        q1, q2, q3, q4 = self.quadrants
        return (q1+q2) - (q3+q4)

    @property
    def x_pos(self) -> float:
        """
        Computes the x position (in arbitrary units).
        The sign is choosen such that a beam on the "right" side leads to positive values.
        Returns
        -------
        float
            x position
        """

        return self.x_diff/self.sum

    @property
    def y_pos(self) -> float:
        """
        Computes the y position (in arbitrary units).
        The sign is choosen such that a beam on the "top" side leads to positive values.
        Returns
        -------
        float
            y position
        """
        return self.y_diff/self.sum

    ## =============================================================================================================

    def dead_region(self, roundoff: float):
        """
        This functions masks out elements of the detector where the gap (between the four segments) exists.
        It returns an array of 1s and 0s over the detector. The elements where the gap is, are 0 ("no light").

        Parameters
        ----------
        roundoff : float
            Scalar fudge factor needed for round-off error

        Returns
        -------
        np.ndarray
            A 2d-array of 1s and 0s over the detector. The elements where the gap is, are 0 ("no light").


        """
        size = self._size
        n = self._n
        gap = self._gap
        delta = size/n  # distance between two elements

        x, y = np.meshgrid(self._detector_x, self._detector_y)

        return (np.abs(x)+delta/2-roundoff>gap/2) & (np.abs(y)+delta/2-roundoff>gap/2)

    def create_detector(self, shape: str, roundoff: float = 1e-14):
        """
        This routine creates the entire detector array.

        Part of this code is adapted from https://github.com/university-of-southern-maine-physics/QuadCellDetector

        For a circular-shaped detector it does so by assuming a
        square array and eliminating chunks not within the circular detector
        boundary.

        Parameters
        ----------
        shape : str
            Defines the shape of the detector. "square": detector is square shaped, "round": detector is circular shaped
        roundoff : float
            Scalar fudge factor needed for round-off error (default = 1e-14)

        Returns
        -------
        array_like
            2d array with effective area of each cell; if the cell is dead,
            it's area will be zero. Most cells will have and area of
            (diameter/n)**2, but some cells which straddle the gap will have a
            fraction of this area.
        """
        size = self._size
        n = self._n
        gap = self._gap

        delta = size/n  # distance between two elements
        self._detector_x = np.linspace(-size/2+delta/2, size/2+delta/2, n)
        self._detector_y = np.linspace(-size/2+delta/2, size/2+delta/2, n)

        x, y = np.meshgrid(self._detector_x, self._detector_y)

        # The maximum possible gap size is sqrt(2)*Radius of detector.
        # raise an exception if this condition is violated.
        # Note: this is only strictly true for a circular shaped detector
        if gap >= np.sqrt(2) * size / 2:
            raise Exception('The gap is too large!')

        if shape == 'circular':
            # This computes the distance of each grid point from the origin
            # and then we extract a masked array of points where r_sqr is less
            # than the distance of each grid point from the origin:
            r_sqr = x**2+y**2
            inside = np.ma.getmask(np.ma.masked_where(r_sqr<=(size/2)**2, x))
        elif shape == 'square':
            inside = np.ones((n,n))
        else:
            raise(ValueError('Invalid shape'))

        dead_region = self.dead_region(roundoff)  # dead region due to gap between the four segments
        self._detector_active = (dead_region*inside).astype(int)

    def plot(self, mode: str):
        """
        2d plot of detector or laser
        Parameters
        ----------
        mode : str
            Defines what to plot. Can be
            * "detector_active": Plot the active region of the detector.
            * "laser_intensity": Plot the intensity of the laser beam
            * "detector_intensity": Plot the intensity of the laser on the detector.
        """
        if mode == 'detector_intensity':
            plt_data = self.detector_intensity
            title = 'Detector Intensity'
        elif mode == 'detector_active':
            plt_data = self.detector_active
            title = 'Detector active'
        elif mode == 'laser_intensity':
            plt_data = self.laser_intensity
            title = 'Laser Intensity'
        else:
            raise(ValueError('Invalid mode'))

        # extent: custom scaling of axes: size of QPD in mm
        extent = [min(self.detector_x), max(self.detector_x), min(self.detector_y), max(self.detector_y)]
        extent = [_ * 1000 for _ in extent]

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        ax.imshow(np.transpose(plt_data), cmap='magma', extent=extent, origin='lower')
        ax.set_xlabel('x position (mm)')
        ax.set_ylabel('y position (mm)')
        ax.set_title(title)
        plt.show()

    def xy_meshgrid(self):
        """
        Convenience function to generate a numpy.meshgrid of the detector array.

        Can e.g. be used to generate the matrix for the laser intensity.

        Returns
        -------
        2d tuple containing two 2d np.ndarrays for x and y coordinate.

        Example
        -------
        Generate Gaussian beam using phytools:
        >>> x, y = qpd.xy_meshgrid()
        >>> qpd.laser_intensity = phytools.functions.gaussian2d(x=x, y=y, a=1, x0=0.8e-3, y0=0, fwhm_x=1e-3, fwhm_y=2e-3, offset=0)
        """
        return np.meshgrid(self._detector_x, self._detector_y, indexing='ij')


