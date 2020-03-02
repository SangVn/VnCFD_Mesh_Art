/**************
* Gmsh tutorial 3
**************/

lc = 0.25;

Point(1) = {0.0, 0.0, 0, lc};
Point(2) = {2.0, 0.0, 0, lc};
Point(3) = {2.0, 1.0, 0, lc};
Point(4) = {0.0, 1.0, 0, lc};

Point(5) = {0.25, 0.25, 0, lc};
Point(6) = {0.75, 0.25, 0, lc};
Point(7) = {0.75, 0.75, 0, lc};
Point(8) = {0.25, 0.75, 0, lc};

Point(9) = {1.25, 0.25, 0, lc};
Point(10) = {1.75, 0.25, 0, lc};
Point(11) = {1.75, 0.75, 0, lc};
Point(12) = {1.25, 0.75, 0, lc};

Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 9};

Curve loop(1) = {4, 1, -2, 3};
Curve loop(2) = {8, 5, 6, 7};
Curve loop(3) = {12, 9, 10, 11};

Plane Surface(1) = {1, 2, 3};
Plane Surface(2) = {2};
Plane Surface(3) = {3};

Physical Surface("out") = {1};
Physical Surface("in1") = {2};
Physical Surface("in2") = {3};

