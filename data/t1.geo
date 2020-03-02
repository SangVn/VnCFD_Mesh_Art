/**************
* Gmsh tutorial 1
**************/

lc = 0.25;

Point(1) = {0.0, 0.0, 0, lc};
Point(2) = {1.0, 0.0, 0, lc};
Point(3) = {1.0, 1.0, 0, lc};
Point(4) = {0.0, 1.0, 0, lc};

Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve loop(1) = {4, 1, -2, 3};
Plane Surface(1) = {1};
