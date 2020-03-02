/**************
* Gmsh tutorial 2
**************/

lc = 0.25;

Point(1) = {0.0, 0.0, 0, lc};
Point(2) = {1.0, 0.0, 0, lc};
Point(3) = {1.0, 1.0, 0, lc};
Point(4) = {0.0, 1.0, 0, lc};

Point(5) = {0.25, 0.25, 0, lc};
Point(6) = {0.75, 0.25, 0, lc};
Point(7) = {0.75, 0.75, 0, lc};
Point(8) = {0.25, 0.75, 0, lc};

Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve loop(1) = {4, 1, -2, 3};
Curve loop(2) = {8, 5, 6, 7};
Plane Surface(1) = {1, 2};
