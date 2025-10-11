
更新时间：2025.08.06

数据结构与算法（DSA）研究如何高效组织数据，利用数组、栈和树等数据结构，结合分步解决问题的步骤（即算法）。数据结构负责数据的存储和访问方式，而算法则专注于处理这些数据。

Below is the formula for finding the area of a triangle whose base and height are given is

> Area = 12×base×height21​×base×height

![Area of Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230320114933/2-(1).png)

- If only sides of the triangle are given. Let an equilateral triangle of side 'a' be given then area of Equilateral Triangle is

> 34×a243​​×a2

![Area of Equilateral Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230327133957/4-(2).png)

## Area of Equilateral Triangle Formula Proof

Let's calculate the area for a given equilateral triangle of side ****a****. It is known that the area of a triangle is given as 1/2 × Base × Height.

![Equilateral Triangle's Area Derivation](https://media.geeksforgeeks.org/wp-content/uploads/20230320115216/1-(3).png)

Here the base is a. Let's find the height of this triangle in order to find the area. It can clearly be seen that the height can be found using the [Pythagoras theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/) since it is one of the sides of the [right-angled triangle](https://www.geeksforgeeks.org/maths/right-angled-triangle/).

Applying [Pythagoras' theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/),

h2 + (a/2)2 = a2

⇒ h2 = (3a2/4)

⇒ h = √3a/2

Now the height of this equilateral triangle is known. Now, substitute this value of height into our formula, 

Area = 1/2 × Base × Height 

⇒ Area = 1/2 × a × √3a/2  =√3a2/4 

> ****Area = √3a********2********/4****

## Derivation of Area of Equilateral Triangle using Trigonometry

Suppose the sides of a triangle are given, then the height can be calculated using the sine formula. Let the sides of a triangle ABC be a, b, and the angle corresponding to them be A, B, and C. Now, the height of a triangle is

h = a × Sin B = b × Sin C = c × Sin A 

Now, area of ABC = ½ × a × (b × sin C) 

⇒ area of ABC = ½ × b × (c × sin A)

⇒ area of ABC = ½ × c (a × sin B)

Since it is an equilateral triangle, A = B = C = 60° and a = b = c

⇒ Area = ½ × a × (a × Sin 60°) 

⇒ Area = ½ × a2 × Sin 60°

⇒ Area = ½ × a2 × √3/2 = √3a2/4 

> ****Area of Equilateral Triangle = (√3/4)a********2****

****Read More****

- [****Trigonometry****](https://www.geeksforgeeks.org/maths/math-trigonometry/)

### ****Perimeter of the Equilateral Triangle****

An equilateral triangle is a triangle with all three sides and the perimeter of any figure is the sum of all its sides. So, the perimeter of an equilateral triangle of side of length "a" is given by 

![Perimeter of the Equilateral Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230320115308/3-(1).png)

****Must Read****

- [Perimeter of an equilateral triangle](https://www.geeksforgeeks.org/maths/how-to-find-the-perimeter-of-an-equilateral-triangle/)

## ****Properties of Equilateral Triangle****

An equilateral triangle is one triangle in which all three sides are equal. For an equilateral triangle PQR, PQ = QR = RP. A few important properties of an equilateral triangle are:

- All three sides are equal in an Equilateral Triangle.

- In an equilateral triangle, all three internal angles are equal to each other and their value is 60°.

- For an equilateral triangle, the median, angle bisector, and perpendicular all are the same.

- Ortho-centre and centroid of an equilateral triangle are the same points.

- In an equilateral triangle, there are three lines of symmetry and also 3rd order rotational symmetry as well.

- Area of an equilateral triangle is √3 a2/ 4.

- Perimeter of an equilateral triangle is 3a.

****Must Read****

- [Area of Triangle](https://www.geeksforgeeks.org/maths/area-of-triangle/)
- [Area of Square](https://www.geeksforgeeks.org/maths/area-of-a-square/)
- [Area of Rhombus](https://www.geeksforgeeks.org/maths/area-of-rhombus/)
- [Area of Rectangle](https://www.geeksforgeeks.org/maths/area-of-rectangle/)
- [Area of Parallelogram](https://www.geeksforgeeks.org/maths/area-of-parallelogram/)
- [Area of Circle](https://www.geeksforgeeks.org/maths/area-of-a-circle/)

## Solved Examples on ****Area of Equilateral triangle****

****Example 1: Find the area of the triangle whose all sides measure 4 units.****

****Solution:****

> As given all sides are of equal length hence, we can say that it is an equilateral triangle.
> 
> So we can apply the formula to directly find the area of this triangle.
> 
> Area = √3a2/4 = √3 × 42/4 = 4√3 units2

****Example 2: Find the perimeter of the triangle whose sides are given as 3 cm, 4 cm, and 5 cm.****

****Solution:**** 

> Sum of all the sides of any triangle is the perimeter of triangle
> 
> Hence, the perimeter of this given triangle is (3 + 4 + 5) cm
> 
> i.e. Perimeter is 12 cm

****Example 3: Find the height of the equilateral triangle whose side is 4 cm.****

****Solution:****

> The formula for the height is given by: h = √3a/2 
> 
> h = (√3 × 4)/2 = 2√3 cm
> 
> Hence the height of the triangle is 2√3 cm

****Example 4: Find the perimeter and area of the equilateral triangle whose side is given as 4 cm.****

****Solution:**** 

> Side (s) = 4 cm
> 
> For any equilateral triangle the perimeter is calculated as 3 × s
> 
> Primeter(P) = 3 × 4 = 12 cm
> 
> Area = √3a2/4   
>         = √3(4)2/4  
>         = √3(16) / 4 cm2  
> Area = 4√3 cm2

****Example 5: Find the area of an equilateral triangle when the perimeter is 18 cm.****

****Solution:****

> Perimeter of an equilateral triangle = 18 cm
> 
> Perimeter of the equilateral triangle = 3a
> 
> 3a = 18, a = 6
> 
> The length of side is 6 cm.
> 
> Area, A = √3 a2/ 4 sq units
> 
>             = √3 (6)2/ 4 cm2 ⇒ 36 √3 / 4
> 
>             ****= 9√3 cm********2****
> 
> Then area of the equilateral triangle is ****9√3 cm********2****

- [Equilateral Triangle](https://www.geeksforgeeks.org/maths/area-of-equilateral-triangle/)
- [Equilateral Triangle Area](https://www.geeksforgeeks.org/maths/area-of-equilateral-triangle/)

## Area of Isosceles Triangle

An ****isosceles triangle**** has two equal sides and the angles opposite these equal sides are also equal.

![Areaof-Isosceles-Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20241206155434151920/Areaof-Isosceles-Triangle.webp)

Area of Isosceles Triangle

****Area of Isosceles Triangle Formula :****

> A = ½ × b√(a2 - (b2/4))
> 
> Where,
> 
> a = both the equal sides and b = the third unequal side.

****Example: What is the area of an isosceles triangle with sides 5 cm, 5 cm, and 6 cm?****

****Solution:****

> Using the Formula: A = ½ × b√(a2 - (b2/4))
> 
> - a = 5 cm (the equal sides),
> - b = 6 cm (the base).
> 
> A=3×25−364A=3×25−436​​  
> =3×25−9=3×25−9​  
> =3×16=3×16​

****Learn More :****

- [Area of Isosceles Triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/)
- [Types of Triangle](https://www.geeksforgeeks.org/maths/types-of-triangle/)

## Area of ****Scalene Triangle****

A scalene triangle has ****all three sides of different lengths****, and all three angles are different as well.

****Key Features****:

- No sides are congruent (all sides have different lengths).
- No angles are equal.
- It has no symmetry.

****Example****: A triangle with sides of 3 cm, 5 cm, and 7 cm is a scalene triangle.

## Area of Triangle By Heron's Formula

****The area of a triangle**** ****with 3 sides**** given can be found using Heron's Formula. This formula is useful when the height is not given.

![Area-of-Triangle-using-Herons-Fromula](https://media.geeksforgeeks.org/wp-content/uploads/20241206155424655706/Area-of-Triangle-using-Herons-Fromula.webp)

Area of Triangle using Herons Formula

Heron's Formula is given by,

> ****Area of Triangle = √{s(s - a)(s - b)(s - c)}****
> 
> where, ****a, b****, and ****c**** are sides of the given triangle  
> and ****s**** = ½ (a+b+c) is the semi perimeter.

****Example: What is the area of a triangle with sides of 3 cm, 4 cm, and 5 cm?****

****Solution:****

> Using Heron's formula,  
> s = (a+b+c)/2  
>  = (3+4+5)/2  
>  = 12/2 = 6
> 
> Area = √{ s(s-a)(s-b)(s-c)}  
> = √{ 6(6-3)(6-4)(6-5)}         
> = √(6 × 3 × 2 × 1) = √(36  
> = 6 cm2

****Learn More:**** [****Heron's Formula****](https://www.geeksforgeeks.org/maths/herons-formula/)

## Area of Triangle With Two Sides and Included Angle (SAS)

****Area of SAS Triangle**** is obtained by using the concept of trigonometry.

Let us assume ABC is right angled triangle and AD is perpendicular to BC.

![Area of Triangle in Trigonometry](https://media.geeksforgeeks.org/wp-content/uploads/20230803112308/Triangle-3.png)

In the above figure,

Sin B = AD/AB

⇒ AD = AB Sin B = c Sin B  
⇒ Area of Triangle ABC = 1/2 ⨯ Base ⨯ Height  
⇒ Area of Triangle ABC = 1/2 ⨯ BC ⨯ AD  
⇒ Area of Triangle ABC = 1/2 ⨯ a ⨯ c Sin B  
= 1/2 ⨯ BC ⨯ AD

Thus,

> ****Area of Triangle = 1/2 ac Sin B****

****Similarly,**** we can find that,

> ****Area of Triangle = 1/2 bc Sin A****  
> ****Area of Triangle = 1/2 ab Sin C****

We conclude that the area of a triangle using trigonometry is given as****, half the product of two sides and sine of the included angle.****

## Area of Triangle in Coordinate Geometry

In Coordinate Geometry, if the coordinates of triangle ABC are given as A(x1, y1), B(x2, y2), and C(x3, y3), then its Area is given by the following formula :

![Area-of-trinagle--in-coordinate-geometry](https://media.geeksforgeeks.org/wp-content/uploads/20241206155424772249/Area-of-trinagle--in-coordinate-geometry.webp)

Area of Triangle in coordinate geometry

Area of △ABC = 1/2∣x1y11x2y21x3y31∣1/2∣∣​x1​x2​x3​​y1​y2​y3​​111​∣∣​

> ⇒ Area of △ABC = ½[__x__1​(__y__2 ​− __y__3​) + __x__2​(__y__3 ​− __y__1​) + __x__3​(__y__1 ​− __y__2​)]

****Articles related to the**** Area:

> - [Area of Square](https://www.geeksforgeeks.org/maths/area-of-a-square/)
> - [Area of Rectangle](https://www.geeksforgeeks.org/maths/area-of-rectangle/)
> - [Area of Circle](https://www.geeksforgeeks.org/maths/area-of-a-circle/)
> - [Area of Quadrilateral](https://www.geeksforgeeks.org/maths/area-of-quadrilateral/)
> - [Area of Rhombus](https://www.geeksforgeeks.org/maths/area-of-rhombus/)
> - [Area of Trapezium](https://www.geeksforgeeks.org/maths/area-of-trapezium/)
> - [Area of Parallelogram](https://www.geeksforgeeks.org/maths/area-of-parallelogram/)
> - 
Other than this different formulas are used to find the [area of triangles](https://www.geeksforgeeks.org/maths/area-of-triangle/). Triangles are classified depending on their sides, different types of triangles based on sides are given below:

1. ****Equilateral Triangle:**** Triangle with all three sides equal.
2. ****Isosceles Triangle:**** Triangle with any two sides equal.
3. ****Scalene Triangle:**** Triangle with all sides unequal.

## ****What is an Isosceles Triangle?****

****Isosceles triangle**** is a [type of triangle](https://www.geeksforgeeks.org/maths/properties-of-triangle/) with two equal sides. The two angles opposing the two equal sides are also equal. Assume that in a triangle △ABC, if the sides AB and AC are equal, ABC is an isosceles triangle with ∠B = ∠C. The isosceles triangle is described by the theorem ****"If the two sides of a triangle are equal, then the angle opposite to them are likewise equal".****

![Isosceles triangle](https://media.geeksforgeeks.org/wp-content/uploads/20220829105307/areaofisoscelestriangle.jpg)

Table of Content

- [What is an Isosceles Triangle?](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#what-is-an-isosceles-triangle)
- [Properties of Isosceles triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#properties-of-isosceles-triangle)
- [Formulas for Area of Isosceles Triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#isosceles-triangle-formula)
- [Area of Isosceles Triangle Formula with Sides](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#area-of-isosceles-triangle-formula-with-sides)
- [Derivation for Area of Isosceles Triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#derivation-for-area-of-isosceles-triangle) 
- [Area of Isosceles Triangle Using Heron’s Formula](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#area-of-isosceles-triangle-using-herons-formula)
- [Area of Right Angled Isosceles Triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#area-of-right-angled-isosceles-triangle)
- [Area of Isosceles Triangle using Trigonometry](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#area-of-isosceles-triangle-using-trigonometry)
- [Solved Examples on Area of Isosceles Triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/#solved-examples-on-area-of-isosceles-triangle)

## Properties of Isosceles triangle

- ****Isosceles triangle has two equal sides**** and are known as the legs and the angle between those equal sides is called the vertex angle or apex angle.
- The side opposite the vertex angle is called the base.
- The angle opposite to equal sides are also equal and are known as base angles.
- The perpendicular drawn from the vertex angle bisects the base and the vertex angle.

## ****Formulas for Area of Isosceles Triangle****

Area of an isosceles triangle is given by the formula listed below:

> ****Area = ½ × base × Height****

Also,

> ****Perimeter of isosceles triangle (P) = 2a + b****  
> ****Altitude of isosceles triangle (h) = √(a********2**** ****− b********2********/4)****
> 
> where, ****a, b**** are the sides of a isosceles triangle.

### ****Formulas:****

The following table contains various formulas are used to find the Area of the Isosceles Triangle . Few of the most used formulas for the area of the isosceles triangle are listed below:

|Known Parameters|Formula|
|---|---|
|****base and height/altitude (b and h)****|****½ × b × h****|
|****all three sides ( a, b and c)****|****½[√(a********2**** ****− b********2**** ****⁄4) × b]****|
|****length of 2 sides and an angle between them (b,c and α)****|****½ × b × c × sin(α)****|
|****two angles and the length between (α, β**** ****and c)****|****[c********2********×sin(β)×sin(α)/ 2×sin(2π−α−β)]****|
|****isosceles right triangle (a)****|****½ × a********2****|

## ****Area of Isosceles Triangle Formula with Sides****

When the length of equal sides and the length of the base of an isosceles triangle are given, then the height of the triangle can also be calculated by the given formula:

****Altitude of an Isosceles Triangle = √(a********2**** ****− b********2********/4)****

> ****Area of Isosceles Triangle (if all sides are given) = ½[√(a********2**** ****− b********2**** ****/4) × b]****
> 
> Where,
> 
> - ****b**** = base of the isosceles triangle, and
> - ****a**** = length of the two equal sides.

## ****Derivation for Area of Isosceles Triangle**** 

> If the lengths of an isosceles triangle's equal sides and base are known, the triangle's height or altitude may be computed. The formula for calculating the area of an isosceles triangle with sides is as follows:
> 
> Isosceles triangle area = ****½[√(a********2**** ****− b********2**** ****/4) × b]****
> 
> where,
> 
> ****b**** = the isosceles triangle's base  
> ****a**** = the length of two equal sides

![Derivation for Isosceles Triangle Area](https://media.geeksforgeeks.org/wp-content/uploads/20220831122020/DerivationofAreaofIsoscelestriangle.png)

> From the above figure, we have,
> 
> AB = AC = a (sides of equal length)
> 
> BD = DC = ½ BC = ½ b (Perpendicular from the vertex angle ∠A bisects the base BC)
> 
> Using Pythagoras theorem on ΔABD,
> 
> a2 = (b/2)2 + (AD)2
> 
> AD = a2−b24a2−4b2​​
> 
> The altitude of an isosceles triangle = a2−b24a2−4b2​​
> 
> It is known that the general formula of area of the triangle is, Area = ½ × b × h
> 
> Substituting value for height, we get
> 
> Area of isosceles triangle = ****½[√(a********2**** ****− b********2**** ****/4) × b]****

## Area of Isosceles Triangle Using Heron’s Formula

The area of an isosceles triangle formula can be easily derived using [Heron’s formula](https://www.cuemath.com/herons-formula/) as explained in the following steps. Heron's formula is used to find the area of a triangle when the measurements of its 3 sides are given.

****Derivation:****

The Heron's formula to find the area, A of a triangle whose sides are a,b, and c is:

> A = √s(s-a)(s-b)(s-c)

where,

- a, b, and c are the sides of the triangle.
- s is the semi perimeter of the triangle.

We know that the [perimeter of a triangle](https://www.cuemath.com/measurement/perimeter-of-a-triangle/) with sides a, b, and c is a + b + c. Here, s is half of the perimeter of the triangle, and hence, it is called semi-perimeter.

> Then, the semi-perimeter is:
> 
> s = (a + b + c)/2
> 
> as b = a,
> 
> s = ½(a + a + b)
> 
> ⇒ s = ½(2a + b) = a + (b/2)
> 
> now,
> 
> Area of Isosceles Triangle = √[s(s−a)(s−b)(s−c)]
> 
> Area = √[s (s−a)2 (s−b)]
> 
> Area = (s−a) × √[s (s−b)]
> 
> Substituting the value for “s”
> 
> ⇒ Area = (a + b/2 − a) × √[(a + b/2) × ((a + b/2) − b)]
> 
> ⇒ Area = b/2 × √[(a + b/2) × (a − b/2)]
> 
> Area of isosceles triangle = b/2 × √(a2 − b2/4) square units
> 
> where,
> 
> - b = base of the isosceles triangle
> - a = length of the two equal sides

## ****Area of Right Angled Isosceles Triangle****

Area of an Isosceles Right Triangle is given by the formula 

![Area of Isosceles Right Triangle Formula](https://media.geeksforgeeks.org/wp-content/uploads/20220829110406/IsoscelesrightAngletriangle.jpg)

Formula for Isosceles Right Triangle ****Area= ½ × a********2****

****Derivation:****

> ****Area of an isosceles triangle**** (Area) = ½ ×base × height
> 
> ⇒ Area = ½ × a × a = a2/2

Perimeter of Isosceles Right Triangle ****P = (2+√2)a****

****Derivation:****

> Perimeter of an isosceles right triangle is the sum of all the sides of an isosceles right triangle.
> 
> Let the two equal sides be ****a****. By Pythagoras theorem the unequal side is ****a√2.****
> 
> Perimeter of isosceles right triangle = a+a+a√2  
> ⇒ Perimeter of isosceles right triangle = 2a+a√2  
> ⇒ Perimeter of isosceles right triangle = a(2+√2)  
> ⇒ Perimeter of isosceles right triangle = a(2+√2)

## ****Area of Isosceles Triangle using Trigonometry****

When the Length of the two Sides and the Angle between them are given,

****A = ½ × b × c × sin(α)****

Where,

- ****b, c**** are sides of a given triangle, and
- ****α**** is the angle between them.

When the two angles and sides between them are given,

> ****A = [c********2********×sin(β)×sin(α)/ 2×sin(2π−α−β)]****
> 
> Where,
> 
> - ****c**** is sides of a given triangle, and
> - ****α,**** ****β**** is the angle associated with them.

****Related Articles****

> - [Area of Square](https://www.geeksforgeeks.org/maths/area-of-a-square/)
> - [Area of Rhombus](https://www.geeksforgeeks.org/maths/area-of-rhombus/)
> - [Area of Rectangle](https://www.geeksforgeeks.org/maths/area-of-rectangle/)

## ****Solved Examples on Area of Isosceles Triangle****

****Example 1: Find the area of an isosceles triangle with an**** ****equal side of**** ****13 cm and a**** ****base of**** ****24 cm.****

****Solution:****

> We have, a = 13 and b = 24.
> 
> Area of isosceles triangle is given by, 
> 
> A = 12×(a2−b24)×b21​×(a2−4b2​​)×b
> 
> ⇒ A = 12×(132−2424)×2421​×(132−4242​​)×24
> 
> ⇒ A = 1/2 × 5 × 24
> 
> ⇒ A = 60 cm2

****Example 2: Find the area of an isosceles triangle with an**** ****equal side of**** ****10 cm and a**** ****base of 12 cm.****

****Solution:****

> We have, a = 10 and b = 12.
> 
> Area of isosceles triangle is given by,
> 
> A = 12×(a2−b24)×b21​×(a2−4b2​​)×b
> 
> ⇒ A = 12×(102−1224)×1221​×(102−4122​​)×12
> 
> ⇒ A = 1/2 × 8 × 12
> 
> ⇒ A = 48 cm2

****Example 3: Find the area of an isosceles triangle with an**** ****equal side of**** ****5 cm and a**** ****base of**** ****6 cm.****

****Solution:****

> We have, a = 5 and b = 6.
> 
> Area of isosceles triangle is given by,
> 
> A = 12×(a2−b24)×b21​×(a2−4b2​​)×b
> 
> ⇒ A = 12×(52−624)×621​×(52−462​​)×6
> 
> ⇒ A = 1/2 × 4 × 6
> 
> ⇒ A = 12 cm2

****Example 4: Find the area of an isosceles triangle with an**** ****equal side of**** ****15 cm and a**** ****base of**** ****24 cm.****

****Solution:****

> We have, a = 15 and b = 24.
> 
> Area of isosceles triangle is given by,
> 
> A = 12×(a2−b24)×b21​×(a2−4b2​​)×b
> 
> ⇒ A = 12×(152−2424)×2421​×(152−4242​​)×24
> 
> ⇒ A = 1/2 × 9 × 24
> 
> ⇒ A = 108 cm2

****Example 5: Find the area of an isosceles triangle with an**** ****equal side of**** ****17 cm and**** a ****base of 30 cm.****

****Solution:****

> We have, a = 17 and b = 30.
> 
> Area of isosceles triangle is given by,
> 
> A = 12×(a2−b24)×b21​×(a2−4b2​​)×b
> 
> ⇒ A = 12×(172−3024)×3021​×(172−4302​​)×30
> 
> ⇒ A = 1/2 × 8 × 30
> 
> ⇒ A = 120 cm2

****Example 6: Find the area of an isosceles triangle with an**** ****equal side of**** ****20 cm and a**** ****base of 24 cm.****

****Solution:****

> We have, a = 20 and b = 24.
> 
> Area of isosceles triangle is given by,
> 
> A = 12×(a2−b24)×b21​×(a2−4b2​​)×b
> 
> ⇒ A = 12×(202−2424)×2421​×(202−4242​​)×24
> 
> ⇒ A = 1/2 × 16 × 24
> 
> ⇒ A = 192 cm2

****Example 7: Find the area of an isosceles triangle with an**** ****equal side of**** ****25 cm and a**** ****base of**** ****30 cm.****

****Solution:****

> We have, a = 25 and b = 30.
> 
> Area of isosceles triangle is given by,
> 
> A = 12×(a2−b24)×b21​×(a2−4b2​​)×b
> 
> ⇒ A = 12×(252−3024)×3021​×(252−4302​​)×30
> 
> ⇒ A = 1/2 × 20 × 30
> 
> ⇒ A = 300 cm2

## Conclusion

****Area of isosceles triangle****, as well as the general area of any triangle, can be calculated using a variety of formulas depending on the known parameters. Whether using the basic base-height method, Heron’s formula, or trigonometric approaches, each method provides a reliable way to determine the area based on the ****isosceles triangle’s**** unique properties. Understanding these formulas not only simplifies geometric calculations but also enhances problem-solving skills in various mathematical contexts.

| Related Articles                                                                          |                                                                                                       |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
|                                                                                           | [****Area of Isosceles Triangle****](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/) |
| [****Equilateral Triangle****](https://www.geeksforgeeks.org/maths/equilateral-triangle/) | [****Heron’s Formula****](https://www.geeksforgeeks.org/maths/herons-formula/)                        |


## Angles of Isosceles Triangle

A triangle has three angles so an isosceles triangle also has three angles but an isosceles triangle is a special case as it has two angles of the three angles equal. The angle sum property of the triangle also holds for the Isosceles triangle. Suppose we have an isosceles triangle △ABC where AB = AC, and  ∠B = ∠C. If the unknown angle ∠A is given then we can easily find the other angle of the Isosceles triangle. For example,

****Example: In Isosceles triangle △ABC where ∠B = ∠C and ∠A = 80°. Find other angles.****  
****Solution:****

> We know that, in any triangle △ABC
> 
> ∠A  + ∠B + ∠C  = 180°
> 
> Also, ∠B = ∠C and ∠A = 80°
> 
> Using angle sum property of triangle,
> 
> 80°  + ∠B + ∠B  = 180°
> 
> ⇒ 2∠B = 100°  
> ⇒ ∠B = 50°
> 
> Thus, measure of other two angles of an isosceles triangle is 50°.

## Types of Isosceles Triangles

Isosceles triangles are classified into three types depending on the measures of angles, which include,

- Isosceles Right Triangle
- Isosceles Obtuse Triangle
- Isosceles Acute Triangle

![Examples-of-Triangles-1](https://media.geeksforgeeks.org/wp-content/uploads/20230609152904/Examples-of-Triangles-1.png)

Types of Isosceles Triangle

### ****Isosceles Right Triangle****

An isosceles triangle that has a right angle is called an Isosceles Right triangle. Examples of isosceles right triangles are,

- Triangle with angles 45°, 45° and 90°

### ****Isosceles Obtuse Triangle****

An isosceles triangle in which any one angle is obtuse angles and the other two are acute angles is called an Isosceles Acute triangle. Some examples of isosceles obtuse triangles are,

- Triangle with angles 40°, 40° and 100°
- Triangle with angles 35°, 35° and 110°, etc.

### ****Isosceles Acute Triangle****

An isosceles triangle in which all the angles are acute is called an Isosceles Acute triangle. Some examples of isosceles acute triangles are,

- Triangle with angles 50°, 50° and 80°
- Triangle with angles 65°, 65° and 50°, etc.

## Properties of Isosceles Triangles

The following are some important characteristics of an isosceles triangle:

- An isosceles triangle will always have at least two equal sides and two equal angles.
- In an isosceles triangle, the two sides of equal length are called the legs, and the third side of the triangle is called the base.
- The angle between the legs of an isosceles triangle is called the apex angle or vertex angle.
- The perpendicular drawn from the apex angle bisects the base of the isosceles triangle and the apex angle.
- The perpendicular drawn from the apex angle is also known as the line of symmetry as it divides the isosceles triangle into two congruent triangles.

## Isosceles Triangle Theorem

There are two common theorems related to isosceles triangles i.e.,

- Base Angle Theorem
- Isosceles Triangle Theorem

### Base Angle Theorem

[Angle Theorem](https://www.geeksforgeeks.org/maths/theorem-angle-opposite-to-equal-sides-of-an-isosceles-triangle-are-equal-class-9-maths/) states that,

> ****"If two sides in any isosceles triangle are equal then the angle opposite to them are also equal."****

### Isosceles Triangle Theorem

The converse of the base angle theorem is also true which states that 

> ****"If two angles in any isosceles triangle are equal then the side opposite to them are also equal."****

If we have an Isosceles triangle ABC then, 

> ****AB = AC ⟺ ∠ABC = ∠ACB****

## Isosceles Triangles Formulas

The height, perimeter, and area are the three basic formulas of an isosceles triangle, which are discussed below.

### Perimeter of Isosceles Triangle

The [perimeter](https://www.geeksforgeeks.org/maths/perimeter/) of an isosceles triangle is equal to the sum of its three side lengths. As an isosceles triangle has two equal sides, the perimeter of the isosceles triangle will be (2a + b) units, where "a" is the length of the two equal sides and "b" is the base length.

![Perimeter of an Isosceles Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230529154648/Perimeter-of-isosceles-triangle.PNG)

> ****Perimeter of an Isosceles Triangle = (2a + b) units****

Where,

- ****"a"**** is the length of the two equal sides, and
- ****"b"**** is the base length

****Learn more:**** [****Perimeter of a Triangle****](https://www.geeksforgeeks.org/maths/perimeter-of-a-triangle/)

### Isosceles Triangle Area

The total region bounded by the three sides of a triangle in a two-dimensional plane is known as the [area of a triangle](https://www.geeksforgeeks.org/maths/area-of-triangle/). The area of an isosceles triangle is equal to half the product of its base length and its height.

![Area of Isosceles Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230529154859/Area-of-Iscosceles-triangle.PNG)

> [****Area of an Isosceles Triangle****](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/) ****= ½ × base × height****

- If the three side lengths of an isosceles triangle are given, then its area can be calculated using [Heron's formula](https://www.geeksforgeeks.org/maths/herons-formula/).

> Area of an Isosceles Triangle = 12×b×[a2−(b/2)2]21​×b×[a2−(b/2)2]​

Where,

- ****"a"**** is Length of Two Equal Sides
- ****"b"**** is Base Length

### Isosceles Triangle Altitude

The perpendicular drawn from the apex angle bisects the base of the isosceles triangle and the apex angle. The formula to calculate the height of an isosceles triangle if its side lengths are given is as follows:

> Height of an Isosceles Triangle (h) =[a2−(b/2)2][a2−(b/2)2]​

Where,

- ****"a"**** is Length of Two Equal Sides
- ****"b"**** is Base Length

> ****Related Articles:****
> 
> - [****Types of Triangles****](https://www.geeksforgeeks.org/maths/types-of-triangle/)
> - [****Pythagoras Theorem****](https://www.geeksforgeeks.org/maths/pythagoras-theorem/)
> - [****Equilateral Triangle****](https://www.geeksforgeeks.org/maths/equilateral-triangle/)
> - [****Scalene Triangle****](https://www.geeksforgeeks.org/maths/scalene-triangle/)

## Solved Examples on Isosceles Triangle

****Example 1: Determine the perimeter of an isosceles triangle with equal sides measuring 7 cm and a base length of 10 cm.****  
****Solution:****

> Given,  
> Lengths of equal sides of the triangle (a) = 7 cm  
> Base length (b) = 10 cm
> 
> We know that,  
> Perimeter of an isosceles triangle (P) = 2a + b  
> ⇒ P = 2 × 7 + 10  
> ⇒ P = 14 + 10 = 24 cm
> 
> Thus, the perimeter of the given isosceles triangle is 24 cm.

****Example 2: Determine the area of an isosceles triangle whose base length is 14 cm, and its height is 7.5 cm.****  
****Solution:****

> Given,  
> Height (h) = 7.5 cm  
> Base length (b) = 14 cm
> 
> We know that, Area of an isosceles triangle (A) = ½ × b × h  
> ⇒ A = ½ × 14 × 7.5  
> ⇒ A = ½ × 105   
> ⇒ A = 52.5 sq. cm
> 
> Hence, the area of the given isosceles triangle is 52.5 sq. cm.

****Example 3: Determine the height of an isosceles triangle with equal sides measuring 13 cm and a base length of 10 cm.****  
****Solution:****

> Given,  
> Lengths of equal sides of the triangle (a) = 13 cm  
> Base length (b) = 10 cm
> 
> We know that,
> 
> Height of an isosceles triangle (h) = [a2−(b/2)2][a2−(b/2)2]​  
> ⇒ h=[132−(10/2)2]h=[132−(10/2)2]​  
> ⇒ h=[132−52]h=[132−52]​  
> ⇒ h=[169−25]h=[169−25]​  
> ⇒ h = √144 = 12 cm
> 
> Hence, the height of the given isosceles triangle is 12 cm.

- [Isosceles Triangle](https://www.geeksforgeeks.org/maths/isosceles-triangle/)
- [Equilateral Triangle](https://www.geeksforgeeks.org/maths/equilateral-triangle/)

## Scalene Triangle Types

Scalene triangles are based on the measure of their interior angles. They can be further classified into three categories that are,

- Acute-Angled Scalene Triangle
- Obtuse-Angled Scalene Triangle
- Right-Angled Scalene Triangle

![Types of Scalene Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230609152904/Examples-of-Triangles-2.png)

****Acute-Angled Scalene Triangle:**** An acute-angled scalene triangle is a scalene triangle in which all the interior angles of the triangle are acute angles. I

****Obtuse-Angled Scalene Triangle****: An obtuse-angled scalene triangle is a scalene triangle in which any one of the interior angles of the triangle is an obtuse angle(i.e. its measure is greater than 90°). The other two angles are acute angles.

****Right-Angled Scalene Triangle:**** A right-angled scalene triangle is a scalene triangle in which any one of the interior angles of the triangle is a right angle (i.e. its measure is 90°). The other two angles are acute angles.

## Properties of Scalene Triangle

Key properties of a scalene triangle are,

- All three sides of a scalene triangle are not equal. (for a scalene triangle △ABC AB ≠ BC ≠ CA)
- No angle of the Scalene triangle is equal to one another. (for a scalene triangle △ABC ∠A ≠ ∠B ≠ ∠C)
- Interior angles of a scalene triangle can be either acute, obtuse, or right angle, but some of all its angle is 180 degrees. (for a scalene triangle △ABC ∠A+∠B+∠C = 180°)
- No line of Symmetry exists in the Scalene triangle

## Difference between Scalene, Equilateral and Isosceles Triangles

The main differences between Scalene, Equilateral and Isosceles Triangles are tabulated below:

|[Equilateral Triangle](https://www.geeksforgeeks.org/maths/equilateral-triangle//)|[Isosceles Triangle](https://www.geeksforgeeks.org/maths/isosceles-triangle/)|Scalene Triangle|
|---|---|---|
|In an Equilateral triangle, all three sides of a triangle are equal.|In an Isosceles triangle, any two sides of the triangle are equal.|In a Scalene triangle, no sides of a triangle are equal to each other.|
|All angles in an equilateral triangle are equal they measure 60 degrees each.|Angles opposite to equal sides of an Isosceles triangle are equal.|No two angles are equal in Scalene triangles.|
|The equilateral triangle is shown in the image added below,<br><br>![Equilateral Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230509170059/Equilateral-Triangle-3.png)|The isosceles triangle is shown in the image added below,<br><br>![Isosceles Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230509170047/Isosceles-Triangle-3.png)|The scalene triangle is shown in the image added below,<br><br>![Scalene Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230509170112/Equilateral-Triangle-1.png)|

****Read More:****

- [****Right Angle Formula****](https://www.geeksforgeeks.org/maths/right-angle/)
- [****Area of Triangle****](https://www.geeksforgeeks.org/maths/area-of-triangle/)
- [****Area of Equilateral Triangle****](https://www.geeksforgeeks.org/maths/area-of-equilateral-triangle/)

## ****Perimeter of Scalene Triangle****

[Perimeter](https://www.geeksforgeeks.org/maths/perimeter/) of any figure is the length of its total boundary. So, the perimeter of a scalene triangle is defined as the sum of all of its three sides.

![Scalene Triangle Perimeter Calculation](https://media.geeksforgeeks.org/wp-content/uploads/20220929130402/perimeteroftriangle.png)

From the above figure,

> ****Perimeter = (a + b + c) units****
> 
> Where ****a, b**** and ****c**** are the sides of the triangle.

## ****Area of Scalene Triangle****

[Area](https://www.geeksforgeeks.org/maths/area/) of any figure is the space enclosed inside its boundaries for the scalene triangle area is defined as the total square unit of space occupied by the Scalene triangle.

Area of the scalene triangle depends upon its base and height of it. The image added below shows a scalene triangle with sides a, b and c and height h units.

![Scalene Triangle Area Calculation](https://media.geeksforgeeks.org/wp-content/uploads/20220929125227/areaofscalenetriangle.png)

### When Base and Height are Given

When the base and the height of the scalene triangle is given then its area is calculated using the formula added below,

> ****A = (1/2) × b × h sq. units****
> 
> Where,
> 
> - ****b**** is the base and 
> - ****h**** is the height (altitude) of the triangle.

### When Sides of a Triangle are Given

If the lengths of all three sides of the scalene triangle are given instead of base and height, we calculate the area using [Heron's formula](https://www.geeksforgeeks.org/maths/herons-formula/), which is given by,

> ****A = √(s(s - a)(s - b)(s - c)) sq. units****
> 
> Where,
> 
> - ****s**** denotes the semi-perimeter of the triangle, i.e, ****s = (a + b + c)/2****, and
> - ****a, b,**** and ****c**** denotes the sides of the triangle.

****Read More:****

- [****Types of Triangles****](https://www.geeksforgeeks.org/maths/types-of-triangle/)
- [****Area of an Equilateral Triangle****](https://www.geeksforgeeks.org/maths/area-of-equilateral-triangle/)
- [****Perimeter of a Triangle****](https://www.geeksforgeeks.org/maths/how-to-find-the-perimeter-and-area-of-a-triangle/)

## ****Solved Examples Scalene Triangle****

****Example 1: Find the perimeter of a scalene triangle with side lengths of 10 cm, 15 cm, and 6 cm.****  
****Solution:****

> We have, 
> 
> a = 10  
> b = 15  
> c = 6
> 
> Using the Perimeter Formula 
> 
> Perimeter (P) = (a + b + c)
> 
> ⇒ P = (10 + 15 + 6)  
> ⇒ P  = 31 cm
> 
> Thus, the required perimeter of the triangle is 31 cm.

****Example 2: Find the length of the third side of a scalene triangle with two side lengths of 3 cm and 7 cm and a perimeter of 20 cm.****  
****Solution:****

> We have, 
> 
> a = 3  
> b = 7  
> P = 20
> 
> Using the Perimeter Formula 
> 
> Perimeter (P) = (a + b + c)
> 
> ⇒ P = (a + b + c)  
> ⇒ 20 = (3 + 7 + c)  
> ⇒ 20 = 10 + c  
> ⇒ c = 10 cm
> 
> Thus, the required length of third side of the triangle is 10 cm

****Example 3: Find the area of a scalene triangle with side lengths of 8 cm, 6 cm, and 10 cm.****  
****Solution:****

> We have, 
> 
> a = 8  
> b = 6  
> c = 10
> 
> Semi-Perimeter (s) = (a + b + c)/2
> 
> ⇒ s = (8 + 6 + 10)/2  
> ⇒ s = 24/2  
> ⇒ s = 12 cm
> 
> Using the [Heron's formula](https://www.geeksforgeeks.org/maths/herons-formula/) 
> 
> Area = √(s(s - a)(s - b)(s - c))
> 
> ⇒ A = √(12(12 - 8)(12 - 6)(12 - 10))  
> ⇒ A  = √(12(4)(6)(2))  
> ⇒ A  = √576  
> ⇒ A  = 24 sq. cm
> 
> Thus, the required area of the scalene triangle is 24 cm2

****Example 4: Find the area of a scalene triangle whose base is 20 cm and altitude is 10 cm.****  
****Solution:****

> We have, 
> 
> b = 20   
> h = 10
> 
> Area of Scalene Triangle ****(A) = 1/2 × b × h****
> 
> ⇒ A  = 1/2 × 20 × 10  
> ⇒ A = 100 sq. cm
> 
> Thus, the area of the given scalene triangle is 100 sq. cm.

The perimeter of a triangle is equal to the sum of all sides of a triangle. Any triangle with three unequal sides is known as the [Scalene Triangle](https://www.geeksforgeeks.org/maths/scalene-triangle/#:~:text=Scalene%20triangle%20is%20defined%20as%20a%20triangle%20whose%20all%20three,triangle%20is%20always%20180%C2%B0.). If the sides of a triangle have lengths equal to a, b, and c, then,

![Perimeter of a Scalene Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230315122021/scalene.png)

Perimeter of a Scalene triangle

****Read More:**** [Scalene Triangle](https://www.geeksforgeeks.org/maths/scalene-triangle/)

### Perimeter of an Isosceles Triangle

For an ****isosceles**** triangle, i.e., any triangle with two sides equal, let two equal sides be of length '****b****' units and the length of the unequal side equals '****c****', then, 

![Perimeter of an Isosceles Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230315122044/Isosceles.png)

****Read More:**** [****Isosceles right triangle****](https://www.geeksforgeeks.org/maths/isosceles-triangle/)

### Perimeter of an Equilateral Triangle

For an ****equilateral**** triangle, since all sides are equal in length, thus a = b = c. Hence,

![Perimeter of an Equilateral Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230315122110/Equilateral.png)

****Read More:**** [Equilateral Triangle](https://www.geeksforgeeks.org/maths/equilateral-triangle/)

### Perimeter of a Right Triangle

For a right-angle triangle i.e. the triangle with one angle of 90°. The perimeter is calculated by adding the length of all given sides. The formula to find the perimeter of a right triangle is:

> ![Perimeter of a Right Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230315122136/Right-angle.png)
> 
> where, 
> 
> ****a**** is the perpendicular, ****b**** is the base and ****c**** is the hypotenous of the right-angled triangle.

****Read More:**** [Right Angle Triangle](https://www.geeksforgeeks.org/maths/right-angled-triangle/)

****Related Article****

> - [Pythagoras theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/)
> - [Perimeter of Rectangle](https://www.geeksforgeeks.org/maths/perimeter-of-rectangle/)
> - [Circumference of circle](https://www.geeksforgeeks.org/maths/circumference-of-circle/)
> - [Perimeter of Square](https://www.geeksforgeeks.org/maths/perimeter-of-square/)
> - [Area of Triangle](https://www.geeksforgeeks.org/maths/area-of-triangle/)
> - [Sum of squares](https://www.geeksforgeeks.org/maths/sum-of-squares/)

## ****Perimeter of a Triangle Examples****

****Example 1: If the length of the sides of a triangle is 4cm, 5cm, and 6cm, then what is the perimeter of the triangle?****  
****Solution:****

> Given, the sides of the triangle are 4cm, 5cm, and 6cm. Thus, it is an ****scalene**** triangle.  
> So the perimeter of the triangle  = Sum of sides = 4 + 5 + 6 = 15cm

****Example 2: What is the perimeter of an equilateral triangle whose one side length is 5cm?****  
****Solution:****

> Given that the triangle is an ****equilateral**** triangle, thus all three sides are equal in length.
> 
> Since one side is equal to 5cm, the other two sides will also be equal to 5cm.  
> So, Perimeter = 5 + 5 + 5 = 15cm.

****Example 3: Given the perimeter of an equilateral triangle is 21cm, find the length of its three sides.****  
****Solution:****

> Since, in an ****equilateral**** triangle, all the three sides are equal in length, the perimeter is equal to three times the length of a side.  
> Let's the length of any side be equal to ****'a'**** units. So perimeter is equal to ****'3a'**** units. So, we can write,
> 
> 3a = 21  
> a = 7cm
> 
> Thus, the length of each side is equal to 7cm.

****Example 4:**** ****Find the length of two equal sides of an isosceles triangle if the length of the unequal side is 5cm and the perimeter is 17cm.****  
****Solution:****

> Given, the length of unequal side is 5cm and perimeter is 17cm.
> 
> Since, it is ****isosceles**** triangle, length of other two sides are equal. Let each equal side length be ****'a'**** units.  
> Thus, perimeter = a + a + 5
> 
> Since, perimeter = 17cm, we can write,  
> 17 = a + a + 5  
> 2a + 5 = 17  
> 2a = 12  
> a = 6cm
> 
> Thus, the length of the equal sides of the isosceles triangle is 6cm.

> [Perimeter of Triangle](https://www.geeksforgeeks.org/maths/perimeter-of-a-triangle/) = (a + b + c) units

The perimeter is a linear value with a unit of length. Therefore,

### ****Area of Right Angled Triangle****

Area of a right triangle is the space occupied by the boundaries of the triangle.

Area of a right angle triangle is given below,

> Area of a Right Triangle = (1/2 × base × height) square units.

****Also View:****

> - [Triangle](https://www.geeksforgeeks.org/maths/triangles/)
> - [Area of Triangle](https://www.geeksforgeeks.org/maths/area-of-triangle/)
> - [Scalene Triangle](https://www.geeksforgeeks.org/maths/scalene-triangle/)
> - [Area of Isosceles Triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/)
> - [Heron’s Formula](https://www.geeksforgeeks.org/maths/herons-formula/)

## Derivation of Right Angled Triangle Area Formula

For any right angle triangle, PQR right angled at Q with hypotenuse as, PR

Now if we flip the triangle over its hypotenuse a rectangle is formed which is named PQRS. The image given below shows the rectangle form by flipping the right triangle.

![Right Triangle Area Formula](https://media.geeksforgeeks.org/wp-content/uploads/20230215121221/Right-Triangle-Formula-2-(3).png)

As we know, the area of a rectangle is given as the product of its length and width, i.e. ****Area = length × breadth****

Thus, the area of Rectangle PORS = b x h

Now, the area of the right angle triangle is twice the area of the rectangle then,

Thus,

Area of ∆PQR = 1/2 × Area of Rectangle PQRS

> ****A = 1/2 × b × h****

## Hypotenuse of Right Angled Triangle

For a right triangle, the hypotenuse is calculated using the Pythagoras Theorem,

> ****H = √(P********2**** ****+ B********2********)****

where,

- ****H**** is Hypotenuse of Right Triangle
- ****P**** is Perpendicular of Right Triangle

## Solved Examples Questions

Let's solve some example problems on right angled triangles.

****Example 1: Find the area of a triangle if the height and hypotenuse of a right-angled triangle are 10 cm and 11 cm, respectively.**** 

****Solution:**** 

> Given: 
> 
> - Height = 10 cm
> - Hypotenuse = 11 cm
> 
> Using Pythagoras' theorem,
> 
> (Hypotenuse)2 = (Base)2 + (Perpendicular)2
> 
> (11)2 = (Base)2 + (10)2
> 
> (Base)2 = (11)2 - (10)2 = 121 - 100 
> 
> Base = √21 = 4.6 cm
> 
> Area of the Triangle = (1/2) × b × h
> 
> Area = (1/2) × 4.6 × 10
> 
> Area = 23 cm2

****Example 2: Find out the area of a right-angled triangle whose perimeter is 30 units, height is 8 units, and hypotenuse is 12 units.****

****Solution:****

> - Perimeter = 30 units
> - Hypotenuse = 12 units
> - Height = 8 units
> 
> Perimeter = base + hypotenuse + height
> 
> 30 units = 12 + 8 + base
> 
> Base = 30 - 20 = 10 units
> 
> Area of Triangle = 1/2×b×h = 1/2 ×10 × 8 = 40 sq units

****Example 3: If two sides of a triangle are given find out the third side i.e. if Base = 3 cm and Perpendicular = 4 cm find out the hypotenuse.****

****Solution:****

> Given: 
> 
> - Base (b) = 3 cm 
> - Perpendicular (p) = 4 cm
> - Hypotenuse (h) = ?
> 
> Using Pythagoras theorem,
> 
> (Hypotenuse)2 = (Perpendicular)2 + (Base)2
> 
> = 42 + 32 = 16 + 9 = 25 cm2
> 
> Hypotenuse = √(25)
> 
> ****Hypotenuse**** ****= 5 cm****

****Important Maths Related Links:****

- [Mathematics Ratio And Proportion](https://www.geeksforgeeks.org/maths/ratio-and-proportion/)
- [Mathematics Solution](https://www.geeksforgeeks.org/maths/maths/)
- [Exterior Angle Property](https://www.geeksforgeeks.org/maths/exterior-angle-theorem/)
- [Circles Class 9](https://www.geeksforgeeks.org/maths/ncert-solutions-class-9-maths-chapter-10-circles/)
- [Lines Of Symmetry Worksheet](https://www.geeksforgeeks.org/maths/line-of-symmetry/)
- [Multiplicative Identity](https://www.geeksforgeeks.org/maths/additive-identity-vs-multiplicative-identity/)

## Conclusion

Right triangle is an important shape in geometry, defined by one 90-degree angle. It is widely used in various real-world applications, from building structures to solving distance related problems. The relationship between the sides and angles of a right triangle , especially through the Pythagorean theorem, is essential for calculating lengths and understanding trigonometry.


The three sides of a [right-angled triangle](https://www.geeksforgeeks.org/maths/right-angled-triangle/) are as follows,

![Right-Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20240112163734/Right-Triangle.png)

Right Triangle

- ****Base:**** The side(RQ) on which the angle θ lies is known as the base.

- ****Perpendicular:**** It is the side(PQ) opposite to the angle θ  in consideration.

- ****Hypotenuse:**** It is the longest side(PR) in a right-angled triangle and opposite to the 90° angle.

## Trigonometric Functions

[Trigonometry](https://www.geeksforgeeks.org/maths/math-trigonometry/) has 6 basic trigonometric functions, they are sine, cosine, tangent, cosecant, secant, and cotangent. Now let's look into the trigonometric functions. The six trigonometric functions are as follows,

- [****Sine Function****](https://www.geeksforgeeks.org/maths/sine-function/)****:**** It is represented as sin θ and is defined as the ratio of perpendicular and hypotenuse.

- [****Cosine Function****](https://www.geeksforgeeks.org/maths/cosine-function/)****:**** It is represented as cos θ and is defined as the ratio of base and hypotenuse.

- [****Tangent Function****](https://www.geeksforgeeks.org/maths/tangent-function/)****:**** It is represented as tan θ and is defined as the ratio of sine and cosine of an angle. Thus the definition of tangent comes out to be the ratio of perpendicular and base.

- [****Cosecant Function****](https://www.geeksforgeeks.org/maths/cosecant-formula/)****:**** It is the reciprocal of sin θ and is represented as cosec θ.

- [****Secant Function****](https://www.geeksforgeeks.org/maths/secant-formula-concept-formulae-solved-examples/)****:**** It is the reciprocal of cos θ and is represented as sec θ.

- [****Cotangent Function****](https://www.geeksforgeeks.org/maths/cotangent-formula/)****:**** It is the reciprocal of tan θ and is represented as cot θ.

## What are Six Trigonometry Functions?

The six [trigonometric functions](https://www.geeksforgeeks.org/maths/trigonometric-functions/) have formulae for the right-angled triangles, the formulae help in identifying the lengths of the sides of a right-angled triangle, lets take a look at all those formulae,

|****Trigonometric Functions****|****Formulae****|
|---|---|
|sin θ||
|cos θ||
|tan θ||
|cosec θ||
|sec θ||
|cot θ||

The below table shows the values of these functions at some standard angles,

|****Function****|****0°****|****30°****|****45°****|****60°****|****90°****|
|---|---|---|---|---|---|
|sinθ=PHsinθ=HP​|00|1221​|122​1​|3223​​|11|
|cosθ=BHcosθ=HB​|11|3223​​|122​1​|1221​|00|
|tanθ=sinθcosθ=PBtanθ=cosθsinθ​=BP​|00|133​1​|11|33​|∞|
|cosecθ=HPcosecθ=PH​|∞|22|22​|233​2​|11|
|secθ=HBsecθ=BH​|11|233​2​|22​|22|∞|
|cotθ=BPcotθ=PB​|∞|33​|11|133​1​|00|

> ****Note:**** It is advised to remember the first 3 trigonometric functions and their values at these standard angles for ease of calculations.

## Sample Problems on Six Trigonometric Functions

****Problem 1: Evaluate sine, cosine, and tangent in the following figure****.

![Right-Triangle(3-4-5)](https://media.geeksforgeeks.org/wp-content/uploads/20240509114449/Right-Triangle(3-4-5).webp)

****Solution:**** 

> Given,
> 
> - P = 3
> - B = 4
> - H = 5
> 
> Using the trigonometric formulas for sine, cosine and tangent,
> 
> sinθ=PH=35sinθ=HP​=53​
> 
> cosθ=BH=45cosθ=HB​=54​
> 
> tanθ=PB=34tanθ=BP​=43​

****Problem 2: In the same triangle evaluate secant, cosecant****, ****and cotangent.**** 

****Solution:**** 

> As it is known the values of sine, cosine and tangent, we can easily calculate the required ratios.
> 
> cosecθ=1sinθ=53cosecθ=sinθ1​=35​
> 
> secθ=1cosθ=54secθ=cosθ1​=45​
> 
> cotθ=1tanθ=43cotθ=tanθ1​=34​

****Problem 3: Given**** tanθ=68tanθ=86​****, evaluate sin θ.cos θ.****

****Solution:**** 

> tanθ=PBtanθ=BP​
> 
> Thus P = 6, B = 8
> 
> Using Pythagoras theorem,
> 
> H2 = P2 + B2
> 
> H2= 36 + 64 = 100
> 
> Therefore, H =10
> 
> Now, sinθ=610sinθ=106​
> 
> cosθ=810cosθ=108​

****Problem 4: If**** cotθ=1213cotθ=1312​****, evaluate tan********2********θ.****

****Solution:**** 

> Given cotθ=1213cotθ=1312​
> 
> Thus tanθ=1cotθ=1312tanθ=cotθ1​=1213​
> 
> ∴tan2θ=169144∴tan2θ=144169​

****Problem 5: In the given triangle, verify**** ****sin********2********θ + cos********2********θ = 1****

![Right-Triangle(51213)](https://media.geeksforgeeks.org/wp-content/uploads/20240509114912/Right-Triangle(51213).webp)

****Solution:**** 

> Given,
> 
> - P = 12
> - B = 5
> - H = 13
> 
> Thus sinθ=1213sinθ=1312​
> 
> cosθ=513cosθ=135​
> 
> sin2θ=144/169sin2θ=144/169
> 
> cos2θ=25/169cos2θ=25/169
> 
> sin2θ+cos2θ=169169=1sin2θ+cos2θ=169169​=1
> 
> Hence verified.



Half-angle identities for some popular [trigonometric functions](https://www.geeksforgeeks.org/maths/what-are-the-six-trigonometry-functions/) are,

- Half Angle Formula of Sin,

> ****sin A/2 = ±√[(1 - cos A) / 2]****

- Half Angle Formula of Cos, 

> ****cos A/2 = ±√[(1 + cos A) / 2]****

- Half Angle Formula of Tan,

> ****tan A/2 = ±√[1 - cos A] / [1 + cos A]****
> 
> ****tan A/2 = sin A / (1 + cos A)****
> 
> ****tan A/2 = (1 - cos A) / sin A****

## Half Angle Formulas Derivation Using Double Angle Formulas

Half-angle formulas are derived using double-angle formulas. Before learning about half-angle formulas we must learn about Double-angle in [Trigonometry](https://www.geeksforgeeks.org/maths/math-trigonometry/), The most commonly used double-angle formulas in trigonometry are:

- sin 2x = 2 sin x cos x
- cos 2x = cos2 x - sin2 x  
               = 1 - 2 sin2x  
               = 2 cos2x - 1
- tan 2x = 2 tan x / (1 - tan2x)

Now replacing x with x/2 on both sides in the above formulas we get

- sin x = 2 sin(x/2) cos(x/2)
- cos x = cos2 (x/2) - sin2 (x/2)  
              = 1 - 2 sin2 (x/2)  
              = 2 cos2(x/2) - 1
- tan A = 2 tan (x/2) / [1 - tan2(x/2)]

> ****Read More:**** [Double Angled Formulas](https://www.geeksforgeeks.org/maths/double-angle-formulas/)

### ****Half-Angle Formula for Cos Derivation****

We use cos2x = 2cos2x - 1 to find the Half-Angle Formula for Cos

Put x = 2y in the above formula

cos (2)(y/2) = 2cos2(y/2) - 1

cos y = 2cos2(y/2) - 1

1 + cos y = 2cos2(y/2) 

2cos2(y/2) = 1 + cosy

cos2(y/2) = (1+ cosy)/2

> ****cos(y/2) = ± √{(1+ cosy)/2}****

### ****Half-Angle Formula for Sin Derivation****

We use cos 2x = 1 - 2sin2x for finding the Half-Angle Formula for Sin

Put x = 2y in the above formula

cos (2)(y/2) = 1 - 2sin2(y/2)     

cos y = 1 - 2sin2(y/2)   

2sin2(y/2) = 1 - cosy

sin2(y/2) = (1 - cosy)/2

> ****sin(y/2) = ± √{(1 - cosy)/2}****

### ****Half-Angle Formula for Tan Derivation****

We know that tan x  = sin x / cos x such that,

tan(x/2) = sin(x/2) / cos(x/2)

Putting the values of half angle for sin and cos. We get,

tan(x/2) = ± [(√(1 - cosy)/2 ) / (√(1+ cosy)/2 )]

tan(x/2) = ± [√(1 - cosy)/(1+ cosy) ]

Rationalising the denominator

tan(x/2) = ± (√(1 - cosy)(1 - cosy)/(1+ cosy)(1 - cosy))

tan(x/2) = ± (√(1 - cosy)2/(1 - cos2y))

tan(x/2) = ± [√{(1 - cosy)2/( sin2y)}]

> ****tan(x/2) = (1 - cosy)/( siny)****

****Also, Check****

- [Sum to Product Identities](https://www.geeksforgeeks.org/maths/sum-to-product-formulas/)
- [Product Identities](https://www.geeksforgeeks.org/maths/product-to-sum-formulas/)
- [Sin Cos Formulas](https://www.geeksforgeeks.org/maths/sin-cos-formulas-in-trigonometry-with-examples/)
- [Real-Life Applications of Trigonometry](https://www.geeksforgeeks.org/maths/applications-of-trigonometry/)

## Solved Examples of Half Angle Formulas

****Example 1: Determine the value of sin 15°****

****Solution:****

> We know that the formula for half angle of sine is given by:
> 
> sin x/2 = ± ((1 - cos x)/ 2) 1/2
> 
> The value of sine 15° can be found by substituting x as 30° in the above formula
> 
> sin 30°/2 = ± ((1 - cos 30°)/ 2) 1/2
> 
> sin 15° = ± ((1 - 0.866)/ 2) 1/2
> 
> sin 15° = ± (0.134/ 2) 1/2
> 
> sin 15° = ± (0.067) 1/2
> 
> sin 15° = ± 0.2588

****Example 2: Determine the value of sin 22.5****°

****Solution:****

> We know that the formula for half angle of sine is given by:
> 
> sin x/2 = ± ((1 - cos x)/ 2) 1/2
> 
> The value of sine 15° can be found by substituting x as 45° in the above formula
> 
> sin 45°/2 = ± ((1 - cos 45°)/ 2) 1/2
> 
> sin 22.5° = ± ((1 - 0.707)/ 2) 1/2
> 
> sin 22.5° = ± (0.293/ 2) 1/2
> 
> sin 22.5° = ± (0.146) 1/2
> 
> sin 22.5° = ± 0.382

****Example 3: Determine the value of tan 15°****

****Solution:****

> We know that the formula for half angle of sine is given by:
> 
> tan x/2 = ± (1 - cos x)/ sin x
> 
> The value of tan 15° can be found by substituting x as 30° in the above formula
> 
> tan 30°/2 = ± (1 - cos 30°)/ sin 30°
> 
> tan 15° = ± (1 - 0.866)/ sin 30
> 
> tan 15° = ± (0.134)/ 0.5
> 
> tan 15° = ± 0.268

****Example 4: Determine the value of tan 22.5°****

****Solution:****

> We know that the formula for half angle of sine is given by:
> 
> tan x/2 = ± (1 - cos x)/ sin x
> 
> The value of tan 22.5° can be found by substituting x as 45° in the above formula
> 
> tan 30°/2 = ± (1 - cos 45°)/ sin 45°
> 
> tan 22.5° = ± (1 - 0.707)/ sin 45°
> 
> tan 22.5° = ± (0.293)/ 0.707
> 
> tan 22.5° = ± 0.414

****Example 5: Determine the value of cos 15°****

****Solution:****

> We know that the formula for half angle of sine is given by:
> 
> cos x/2 = ± ((1 + cos x)/ 2) 1/2
> 
> The value of sine 15° can be found by substituting x as 30° in the above formula
> 
> cos 30°/2 = ± ((1 + cos 30°)/ 2) 1/2
> 
> cos 15° = ± ((1 + 0.866)/ 2) 1/2
> 
> cos 15° = ± (1.866/ 2) 1/2
> 
> cos 15° = ± (0.933) 1/2
> 
> cos 15° = ± 0.965

****Example 6: Determine the value of cos 22.5°****

****Solution:****

> We know that the formula for half angle of sine is given by:
> 
> cos x/2 = ± ((1 + cos x)/ 2) 1/2
> 
> The value of sine 15° can be found by substituting x as 45° in the above formula
> 
> cos 45°/2 = ± ((1 + cos 45°)/ 2) 1/2
> 
> cos 22.5° = ± ((1 + 0.707)/ 2) 1/2
> 
> cos 22.5° = ± (1.707/ 2) 1/2
> 
> cos 22.5° = ± ( 0.853 ) 1/2
> 
> cos 22.5° = ± 0.923




- ****Angle Relationships:**** These formulas relate the trigonometric ratios of different angles, such as sum and difference formulas, double angle formulas, and [half angle formulas](https://www.geeksforgeeks.org/maths/half-angle-formula/).
- ****Reciprocal Identities:**** These formulas express one trigonometric ratio in terms of another, such as sin(θ) = 1/cos(θ).
- ****Unit Circle:**** The unit circle is a graphical representation of the trigonometric ratios, and it can be used to derive many other formulas.
- ****Law of Sines and Law of Cosines:**** These laws relate the sides and angles of any triangle, not just right triangles.

➣ Learn all about Trigonometry from basics to advanced through this Trigonometry tutorial- [[Read here!](https://www.geeksforgeeks.org/maths/math-trigonometry//)]

Let's learn about these formulas in detail.

## Basic Trigonometric Ratios

There are 6 ratios in trigonometry. These are referred to as Trigonometric Functions. Below is the list of [trigonometric ratios](https://www.geeksforgeeks.org/maths/trigonometric-ratios/), including sine, cosine, secant, cosecant, tangent, and cotangent.

|List of Trigonometric Ratios|   |
|---|---|
|****Trigonometric Ratio****|****Definition****|
|---|---|
|sin θ|Perpendicular / Hypotenuse|
|cos θ|Base / Hypotenuse|
|tan θ|Perpendicular / Base|
|sec θ|Hypotenuse / Base|
|cosec θ|Hypotenuse / Perpendicular|
|cot θ|Base / Perpendicular|

Easy Way to Remember Trigonometric Ratio: [[****SOHCAHTOA****](https://www.geeksforgeeks.org/maths/sohcahtoa/)]  
_****{S****___illy__ ****O****__wls__ ****H****__ide__ ****C****__ake__ ****A****__nd__ ****H****__oney__ ****T****__ill__ ****O****__ctober__ ****A****__rrives}__

### Unit Circle Formula in Trigonometry

For a unit circle, for which the radius is equal to 1, ****θ**** is the angle. The values of the hypotenuse and base are equal to the radius of the unit circle.

Hypotenuse = Adjacent Side (Base) = 1

The ratios of trigonometry are given by:

> - ****sin θ = y/1 = y****
> - ****cos θ = x/1 = x****
> - ****tan θ = y/x****
> - ****cot θ = x/y****
> - ****sec θ = 1/x****
> - ****cosec θ = 1/y****

![Trigonometry Unit Circle Formula Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20230428164928/Trigonometry-2-(2).webp)

Trigonometric Functions Diagram

## Trigonometric Identities

The relationship between trigonometric functions is expressed via trigonometric identities, sometimes referred to as trig identities or trig formulae. They remain true for all real number values of the assigned variables in them.

- [Reciprocal Identities](https://www.geeksforgeeks.org/maths/reciprocal-of-trigonometric-ratios/)
- [Pythagorean Identities](https://www.geeksforgeeks.org/maths/pythagorean-identities/)
- Periodicity Identities (in Radians)
- [Even and Odd Angle Formula](https://www.geeksforgeeks.org/maths/even-odd-identities/)
- [Cofunction Identities (in Degrees)](https://www.geeksforgeeks.org/maths/cofunction-formulas/)
- [Sum and Difference Identities](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/)
- [Double Angle Identities](https://www.geeksforgeeks.org/maths/double-angle-formulas/)
- [Inverse Trigonometry Formulas](https://www.geeksforgeeks.org/maths/inverse-trigonometric-identities/)
- [Triple Angle Identities](https://www.geeksforgeeks.org/maths/triple-angle-formulas/)
- [Half Angle Identities](https://www.geeksforgeeks.org/maths/half-angle-formula/)
- [Sum to Product Identities](https://www.geeksforgeeks.org/maths/sum-to-product-formulas/)
- [Product Identities](https://www.geeksforgeeks.org/maths/product-to-sum-formulas/)

Let's discuss these identities in detail.

### Reciprocal Identities

All of the reciprocal identities are obtained using a right-angled triangle as a reference. Reciprocal Identities are as follows:

> - ****cosec θ = 1/sin θ****
> - ****sec θ = 1/cos θ****
> - ****cot θ = 1/tan θ****
> - ****sin θ = 1/cosec θ****
> - ****cos θ = 1/sec θ****
> - ****tan θ = 1/cot θ****

### ****Pythagorean Identities****

According to the Pythagoras theorem, in a right triangle, if 'c' is the hypotenuse and 'a' and 'b' are the two legs, then c2 = a2 + b2. We can obtain Pythagorean identities using this theorem and trigonometric ratios. We use these identities to convert one trig ratio into other****.****

> - sin2θ + cos2θ = 1
> - 1 + tan2θ = sec2θ
> - 1 + cot2θ = cosec2θ

![Trigonometry Ratio Chart](https://media.geeksforgeeks.org/wp-content/uploads/20230428160845/Trigonometry-4.webp)

Trigonometry Formulas Chart

### Periodicity Identities (in Radians)

These identities can be used to shift the angles by π/2, π, 2π, etc. These are also known as co-function identities.

![Trigonometric Functions in Four Quadrants](https://media.geeksforgeeks.org/wp-content/uploads/20230428160846/Trigonometry-5.webp)

All [trigonometric identities](https://www.geeksforgeeks.org/maths/trigonometric-identities/) repeat themselves after a particular period. Hence are cyclic in nature. This period for the repetition of values is different for different trigonometric identities. 

> - ****sin (π/2 – A) = cos A & cos (π/2 – A) = sin A****
> - ****sin (π/2 + A) = cos A & cos (π/2 + A) = – sin A****
> - ****sin (3π/2 – A)  = – cos A & cos (3π/2 – A)  = – sin A****
> - ****sin (3π/2 + A) = – cos A & cos (3π/2 + A) = sin A****
> - ****sin (π – A) = sin A &  cos (π – A) = – cos A****
> - ****sin (π + A) = – sin A & cos (π + A) = – cos A****
> - ****sin (2π – A) = – sin A & cos (2π – A) = cos A****
> - ****sin (2π + A) = sin A & cos (2π + A) = cos A****

Here's a table that compares the trigonometric properties in different quadrants :

|****Quadrant****|****Sine (sin θ)****|****Cosine (cos θ)****|****Tangent (tan θ)****|****Cosecant (csc θ)****|****Secant (sec θ)****|****Cotangent (cot θ)****|
|---|---|---|---|---|---|---|
|I (0° to 90°)|Positive|Positive|Positive|Positive|Positive|Positive|
|II (90° to 180°)|Positive|Negative|Negative|Positive|Negative|Negative|
|III (180° to 270°)|Negative|Negative|Positive|Negative|Negative|Positive|
|IV (270° to 360°)|Negative|Positive|Negative|Negative|Positive|Negative|

### Even and Odd Angle Formula

The Even and Odd Angle Formulas , also known as Even-Odd Identities are used to express trigonometric functions of negative angles in terms of positive angles. These trigonometric formulas are based on the properties of even and odd functions.

> - ****sin(-θ) = -sinθ****
> - ****cos(-θ) = cosθ****
> - ****tan(-θ) = -tanθ****
> - ****cot(-θ) = -cotθ****
> - ****sec(-θ) = secθ****
> - ****cosec(-θ) = -cosecθ****

### Cofunction identities (in Degrees)

Cofunction identities give us the interrelationship between various trigonometry functions. The co-function are listed here in degrees:

> - ****sin(90°−x) = cos x****
> - ****cos(90°−x) = sin x****
> - ****tan(90°−x) = cot x****
> - ****cot(90°−x) = tan x****
> - ****sec(90°−x) = cosec x****
> - ****cosec(90°−x) = sec x****

### Sum and Difference Identities

The sum and difference identities are the formulas that relate the sine, cosine, and tangent of the sum or difference of two angles to the sines, cosines, and tangents of the individual angles.

> - ****sin(x+y) = sin(x)cos(y) + cos(x)sin(y)****
> - ****sin(x-y) = sin(x)cos(y) - cos(x)sin(y)****
> - ****cos(x+y) = cos(x)cos(y) - sin(x)sin(y)****
> - ****cos(x-y)=cos(x)cos(y) + sin(x)sin(y)****
> - tan(x+y)=tan x+tan y1−tan x.tan ytan(x+y)=1−tan x.tan ytan x+tan y​
> - tan(x−y)=tan x−tan y1+tan x.tan ytan(x−y)=1+tan x.tan ytan x−tan y​

### Double Angle Identities

Double angle identities are the formulas that express trigonometric functions of angles which are double the measure of a given angle in terms of the trigonometric functions of the original angle.

> - ****sin (2x) = 2sin(x) • cos(x) = [2tan x/(1 + tan********2**** ****x)]****
> - ****cos (2x) = cos********2********(x) - sin********2********(x) = [(1 - tan********2**** ****x)/(1 + tan********2**** ****x)] = 2cos********2********(x) - 1 = 1 - 2sin********2********(x)****
> - ****tan (2x) = [2tan(x)]/ [1 - tan********2********(x)]****
> - ****sec (2x) = sec********2**** ****x/(2 - sec********2**** ****x)****
> - ****cosec (2x) = (sec x • cosec x)/2****

### Inverse Trigonometry Formulas

Inverse trigonometry formulas relate to the inverse trigonometric functions, which are the inverses of the basic trigonometric functions. These formulas are used to find the angle that corresponds to a given trigonometric ratio.

> - ****sin********-1**** ****(–x) = – sin********-1**** ****x****
> - ****cos********-1**** ****(–x) = π – cos********-1**** ****x****
> - ****tan********-1**** ****(–x) = – tan********-1**** ****x****
> - ****cosec********-1**** ****(–x) = – cosec********-1**** ****x****
> - ****sec********-1**** ****(–x) = π – sec********-1**** ****x****
> - ****cot********-1**** ****(–x) = π – cot********-1**** ****x****

### Triple Angle Identities

Triple Angle Identities are formulas used to express trigonometric functions of triple angles (3θ) in terms of the functions of single angles (θ). These trigonometric formulas are useful for simplifying and solving trigonometric equations where triple angles are involved.

> ****sin 3x=3sin x - 4sin********3********x****
> 
> ****cos 3x=4cos********3********x - 3cos x****
> 
> tan⁡ 3x=3tan x−tan3x1−3tan2xtan 3x=1−3tan2x3tan x−tan3x​

### Half Angle Identities

Half-angle identities are those trigonometric formulas that are used to find the sine, cosine, or tangent of half of a given angle. These formulas are used to express trigonometric functions of half-angles in terms of the original angle.

> - sin⁡x2=±1−cos x2sin2x​=±21−cos x​​
> - cosx2=±1+cos x2cos2x​=±21+cos x​​
> - tan⁡(x2)=±1−cos(x)1+cos(x)tan(2x​)=±1+cos(x)1−cos(x)​​
> 
> Also,
> 
> - tan⁡(x2)=±1−cos(x)1+cos(x)tan(2x​)=±1+cos(x)1−cos(x)​​
> - tan⁡(x2)=±(1−cos(x))(1−cos(x))(1+cos(x))(1−cos(x)) tan(2x​)=±(1+cos(x))(1−cos(x))(1−cos(x))(1−cos(x))​​ 
> - =(1−cos(x))21−cos2(x) =1−cos2(x)(1−cos(x))2​​ 
> - =(1−cos(x))2sin2(x) =sin2(x)(1−cos(x))2​​ 
> - =1−cos(x)sin(x)=sin(x)1−cos(x)​
> - tan⁡(x2)=1−cos(x)sin(x)tan(2x​)=sin(x)1−cos(x)​

### Sum to Product Identities

Sum to Product identities are the trigonometric formulas that help us to express sums or differences of trigonometric functions as products of trigonometric functions.

> - ****sinx + siny = 2[sin((x + y)/2)cos((x − y)/2)]****
> - ****sinx − siny = 2[cos((x + y)/2)sin((x − y)/2)]****
> - ****cosx + cosy = 2[cos((x + y)/2)cos((x − y)/2)]****
> - ****cosx − cosy = −2[sin((x + y)/2)sin((x − y)/2)]****

### Product Identities

Product identities, also known as product-to-sum identities are the formulas that allow the expression of products of trigonometric functions as sums or differences of trigonometric functions.

These trigonometric formulas are derived from the sum and difference formulas for sine and cosine.

> - ****sinx⋅cosy = [sin(x + y) + sin(x − y)]/2****
> - ****cosx⋅cosy = [cos(x + y) + cos(x − y)]/2****
> - ****sinx⋅siny = [cos(x − y) − cos(x + y)]/2****

****Next Article:**** [****Trigonometric Functions****](https://www.geeksforgeeks.org/maths/trigonometric-functions/)

## ****Summary****

The following illustration represents all the key trigonometric identities essential for solving any trigonometric problem.

![Trigonometry-Identities](https://media.geeksforgeeks.org/wp-content/uploads/20240216235033/Trigonometry-Identities.webp)

List of all Important Trigonometric Identities

  

## Solved Questions on Trigonometry Formulas

Here are some solved examples on trigonometry formulas to help you get a better grasp of the concepts.

****Question 1: If cosec θ + cot θ = x, find the value of cosec θ - cot θ, using the trigonometry formula.****

****Solution:****

> cosec θ + cot θ = x
> 
> We know that cosec2θ+ cot2θ = 1
> 
> ⇒ (cosec θ -cot θ)( cosec θ+ cot θ) = 1
> 
> ⇒ (cosec θ -cot θ) x = 1
> 
> ⇒ cosec θ -cot θ = 1/x

****Question 2: Using trigonometry formulas, show that tan 10° tan 15° tan 75° tan 80° =1****

****Solution:****

> We have, 
> 
> L.H.S. = tan 10****°**** tan 15****°**** tan 75****°**** tan 80****°****
> 
> ⇒ L.H.S = tan(90-80)****°**** tan 15****°**** tan(90-15)****°**** tan 80****°****
> 
> ⇒ L.H.S = cot 80****°**** tan 15****°**** cot 15****°**** tan 80****°****
> 
> ⇒ L.H.S =(cot 80****°**** * tan 80****°****)( cot 15****°**** * tan 15****°****)
> 
> ⇒ L.H.S = 1 = R.H.S

****Question 3: If sin θ cos θ = 8, find the value of (sin θ + cos θ)********2**** ****using the trigonometry Formulas.****

****Solution:****

> (sin θ + cos θ)2
> 
> = sin2θ + cos2θ + 2sinθcosθ
> 
> = (1) + 2(8) = 1 + 16 = 17
> 
> = (sin θ + cos θ)2 = 17

****Question 4: With the help of trigonometric formulas, prove that (tan θ + sec θ - 1)/(tan θ - sec θ + 1) = (1 + sin θ)/cos θ.****

****Solution:****

> L.H.S = (tan θ + sec θ - 1)/(tan θ - sec θ + 1)
> 
> ⇒ L.H.S = [(tan θ + sec θ) - (sec2θ - tan2θ)]/(tan θ - sec θ + 1), [Since, sec2θ - tan2θ = 1]
> 
> ⇒ L.H.S = {(tan θ + sec θ) - (sec θ + tan θ) (sec θ - tan θ)}/(tan θ - sec θ + 1)
> 
> ⇒ L.H.S = {(tan θ + sec θ) (1 - sec θ + tan θ)}/(tan θ - sec θ + 1)
> 
> ⇒ L.H.S = {(tan θ + sec θ) (tan θ - sec θ + 1)}/(tan θ - sec θ + 1)
> 
> ⇒ L.H.S = tan θ + sec θ
> 
> ⇒ L.H.S = (sin θ/cos θ) + (1/cos θ)
> 
> ⇒ L.H.S = (sin θ + 1)/cos θ
> 
> ⇒ L.H.S = (1 + sin θ)/cos θ = R.H.S. Proved.

  

| Related Articles                                                                              |                                                                                                                                         |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [****Basic Trigonometry Concepts****](https://www.geeksforgeeks.org/maths/math-trigonometry/) | [****Domain and Range of Trigonometric Functions****](https://www.geeksforgeeks.org/maths/domain-and-range-of-trigonometric-functions/) |
| [****Trigonometry Table****](https://www.geeksforgeeks.org/maths/trigonometry-table/)         | [****Applications of Trigonometry****](https://www.geeksforgeeks.org/maths/applications-of-trigonometry/)                               |
## List of Trigonometric Identities

There are a lot of identities in the study of [Trigonometry](https://www.geeksforgeeks.org/maths/trigonometry-formulas/), which involve all the trigonometric ratios. These identities are used to solve various problems throughout the academic landscape as well as the real life. Let us learn all the fundamental and advanced trigonometric identities.

### ****Reciprocal Trigonometric Identities****

In all trigonometric ratios, there is a reciprocal relation between a pair of ratios, which is given as follows:

> - sin θ = 1/cosec θ
> - cosec θ = 1/sin θ  
>      
> - cos θ = 1/sec θ 
> - sec θ = 1/cos θ  
>      
> - tan θ = 1/cot θ
> - cot θ = 1/tan θ

### Pythagorean Trigonometric Identities

[****Pythagorean Trigonometric Identities****](https://www.geeksforgeeks.org/maths/pythagorean-identities/) are based on the Right-Triangle theorem or [Pythagoras theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/), and are as follows:

> - sin2 θ + cos2 θ = 1
> - 1 + tan2 θ = sec2 θ
> - cosec2 θ = 1 + cot2 θ

### Trigonometric Ratio Identities

As tan and cot are defined as the ratio of sin and cos, which is given by the following identities:

> - tan θ = sin θ/cos θ
> - cot θ = cos θ/sin θ

### Trigonometric Identities of Opposite Angles

In trigonometry, angles measured in the clockwise direction are measured in negative parity, and all trigonometric ratios defined for negative parity of angle are defined as follows:

> - sin (-θ) = -sin θ
> - cos (-θ) = cos θ
> - tan (-θ) = -tan θ
> - cot (-θ) = -cot θ
> - sec (-θ) = sec θ
> - cosec (-θ) = -cosec θ

### Complementary Angles Identities

[****Complementary angles****](https://www.geeksforgeeks.org/maths/complementary-angles/) are a pair of angles whose measures add up to 90°. Now, the trigonometric identities for complementary angles are as follows:

> - sin (90° – θ) = cos θ
> - cos (90° – θ) = sin θ
> - tan (90° – θ) = cot θ
> - cot (90° – θ) = tan θ
> - sec (90° – θ) = cosec θ
> - cosec (90° – θ) = sec θ

### Supplementary Angles Identities

Supplementary angles are a pair of angles whose measures add up to 180°. Now, the trigonometric identities for supplementary angles are:

> - sin (180°- θ) = sinθ
> - cos (180°- θ) = -cos θ
> - cosec (180°- θ) = cosec θ
> - sec (180°- θ)= -sec θ
> - tan (180°- θ) = -tan θ
> - cot (180°- θ) = -cot θ

### Periodicity of Trigonometric Function

[****Trigonometric functions****](https://www.geeksforgeeks.org/maths/trigonometric-functions/)such as sin, cos, tan, cot, sec, and cosec are all periodic and have different periodicities. The following identities for the trigonometric ratios explain their periodicity.

> - sin (n × 360° + θ) = sin θ
> - sin (2nπ + θ) = sin θ  
>      
> - cos (n × 360° + θ) = cos θ
> - cos (2nπ + θ) = cos θ  
>      
> - tan (n × 180° + θ) = tan θ
> - tan (nπ + θ) = tan θ  
>      
> - cosec (n × 360° + θ) = cosec θ
> - cosec (2nπ + θ) = cosec θ  
>      
> - sec (n × 360° + θ) = sec θ
> - sec (2nπ + θ) = sec θ  
>      
> - cot (n × 180° + θ) = cot θ
> - cot (nπ + θ) = cot θ
> 
> Where, n ∈ ****Z,**** (Z = set of all integers)
> 
> ****Note:**** sin, cos, cosec, and sec have a period of 360° or 2π radians, and for tan and cot period is 180° or π radians.

### Sum and Difference Identities

Trigonometric identities for the [Sum and Difference of angles](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/) include formulas such as sin(A+B), cos(A-B), tan(A+B), etc.

> - sin (A+B) = sin A cos B + cos A sin B
> - sin (A-B) = sin A cos B - cos A sin B
> - cos (A+B) = cos A cos B - sin A sin B
> - cos (A-B) = cos A cos B + sin A sin B
> - tan (A+B) = (tan A + tan B)/(1 - tan A tan B)
> - tan (A-B) = (tan A - tan B)/(1 + tan A tan B) 

****Note:**** Identities for sin (A+B), sin (A-B), cos (A+B), and cos (A-B) are called ****Ptolemy’s Identities****.

### Double Angle Identities

Using the trigonometric identities of the sum of angles, we can find a new identity, which is called the [****Double Angle Identities****](https://www.geeksforgeeks.org/maths/double-angle-formulas/). To find these identities, we can put A = B in the sum of angle identities. For example,

> a  we know, sin (A+B) = sin A cos B + cos A sin B
> 
> Substitute A = B = θ on both sides here, and we get:
> 
> sin (θ + θ) = sinθ cosθ + cosθ sinθ
> 
> - ****sin 2θ = 2 sinθ cosθ****
> 
> Similarly,
> 
> - ****cos 2θ = cos********2********θ - sin**** ****2********θ = 2 cos**** ****2**** ****θ - 1 = 1 - 2sin**** ****2**** ****θ****
> - ****tan 2θ = (2tanθ)/(1 - tan********2********θ)****

### Half-Angle Formulas

Using double-angle formulas, [half-angle formulas](https://www.geeksforgeeks.org/maths/half-angle-formula/) can be calculated. To calculate ****the angle formula,**** replace θ with θ/2, then,

![half-angle](https://media.geeksforgeeks.org/wp-content/uploads/20250407110559723965/half-angle.png)

Half Angle formulas

Other than the above-mentioned identities, there are some more half-angle identities, which are as follows:

![file](https://media.geeksforgeeks.org/wp-content/uploads/20250407110929327715/file.png)

Other half-Angle identities

### Product-Sum Identities

The following identities state the relationship between the sum of two trigonometric ratios with the product of two trigonometric ratios.

![HH](https://media.geeksforgeeks.org/wp-content/uploads/20250407111839406670/HH.png)

Product Sum Identities

### Products Identities

Product Identities are formed when we add two of the sum and difference of angle identities and are as follows:

![file](https://media.geeksforgeeks.org/wp-content/uploads/20250407112628238759/file.png)

Product identities

### Triple Angle Formulas

Other than double and half-angle formulas, there are identities for trigonometric ratios that are defined for triple angles. These [****triple-angle identities****](https://www.geeksforgeeks.org/maths/triple-angle-formulas/) are as follows:

![file](https://media.geeksforgeeks.org/wp-content/uploads/20250407113103332004/file.png)

Triple Angle Formulas

## ****Proof of the Trigonometric Identities****

For any acute angle θ, prove that 

> 1. tanθ = sinθ/cosθ
> 2. cotθ = cosθ/sinθ
> 3. tanθ . cotθ = 1
> 4. sin2θ + cos2θ = 1
> 5. 1 + tan2θ = sec2θ
> 6. 1 + cot2θ = cosec2θ

****Proof:****

> Consider a right-angled △ABC in which ∠B = 90°
> 
> Let AB = x units, BC = y units and AC = r units. 
> 
> ![Right Angle Triangle with Acute Angle Theta](https://media.geeksforgeeks.org/wp-content/uploads/20220927183328/rightangleratios.png)
> 
> Then, 
> 
> ****(1)**** tanθ = P/B = y/x = (y/r) / (x/r) 
> 
> ****∴ tanθ = sinθ/cosθ**** 
> 
> ****(2)**** cotθ = B/P = x/y = (x/r) / (y/r)
> 
> ****∴ cotθ = cosθ/sinθ**** 
> 
> ****(3)**** tanθ . cotθ  = (sinθ/cosθ) . (cosθ/sinθ) 
> 
> ****tanθ . cotθ = 1**** 
> 
> Then, by Pythagoras' theorem, we have 
> 
> x2 + y2 = r2. 
> 
> Now, 
> 
> ****(4)**** sin2θ + cos2θ  = (y/r)2 + (x/r)2 = ( y2/r2 + x2/r2)
> 
>                               = (x2 + y2)/r2 = r2/r2 = 1 [x2+ y2 = r2]
> 
> ****sin********2********θ + cos********2********θ = 1****
> 
> ****(5)**** 1 + tan2θ = 1 + (y/x)2 = 1 + y2/x2 = (y2 + x2)/x2 = r2/x2 [x2 + y2 = r2]
> 
> (r/x)2 = sec2θ 
> 
> ****∴ 1 + tan********2********θ = sec********2********θ.****
> 
> ****(6)**** 1 + cot2θ = 1 + (x/y)2 = 1 + x2/y2 = (x2 + y2)/y2 = r2/y2 [x2 + y2 = r2]
> 
> (r2/y2) = cosec2θ
> 
> ****∴ 1 + cot********2********θ = cosec********2********θ****

## Relation between Angles and Sides of a Triangle

The three rules that relate the sides of triangles to the interior angles are:

- Sine Rule
- Cosine Rule
- Tangent Rule

If a triangle ABC with sides a, b, and c, which are opposite sides to ∠A, ∠B, and ∠C, respectively, then

### Sine Rule

[Sine rule](https://www.geeksforgeeks.org/maths/sine-rule/) states the relationship between sides and angles of the triangle, which is the ratio of the side and the sine of the angle opposite to the side, always remains the same for all the angles and sides of the triangle, and is given as follows:

![file](https://media.geeksforgeeks.org/wp-content/uploads/20250407123440298883/file.png)

Sine Rule

### Cosine Rule 

[Cosine Rule](https://www.geeksforgeeks.org/maths/what-are-cosine-formulas/) involves all the sides, and one interior angle of the triangle is given as follows:

![file](https://media.geeksforgeeks.org/wp-content/uploads/20250407113651237751/file.png)

Cosine Rule

### Tangent Rule

- Tangent Rule also states the relationship between the sides and the interior angle of a triangle, using the tangent trigonometric ratio, which is as follows:

![file](https://media.geeksforgeeks.org/wp-content/uploads/20250407114302134381/file.png)

Tangent Rule

****Also, Read****

> - [Trigonometric Table](https://www.geeksforgeeks.org/maths/trigonometry-table/)
> - [Trigonometry Height and Distance](https://www.geeksforgeeks.org/maths/height-and-distance/)

## Summarizing Trigonometric Identities

  

![Trigonometry-Identities](https://media.geeksforgeeks.org/wp-content/uploads/20240216235033/Trigonometry-Identities.webp)

Important Trigonometric Identities

  

## Trigonometric Identities Examples

****Example 1: Prove that (1 - sin********2********θ) sec********2********θ = 1****  

****Solution:****

> We have: 
> 
> LHS = (1 - sin2θ) sec2θ  
> = cos2θ . sec2θ   
> = cos2θ . (1/cos2θ)  
> = 1   
> = RHS
> 
> ∴ LHS = RHS. [Hence Proved]

****Example 2: Prove that (1 + tan********2********θ) cos********2********θ = 1****  

****Solution:****

> We have:
> 
> LHS = (1 + tan2θ)cos2θ  
> ⇒ LHS = sec2θ . cos2θ  
> ⇒ LHS = (1/cos2θ) . cos2θ  
> ⇒ LHS = 1 = RHS
> 
> ∴ LHS = RHS [Hence Proved]

****Example 3: Prove that (cosec********2********θ - 1) tan²θ = 1**** 

****Solution:****

> We have: 
> 
> LHS = (cosec²θ - 1) tan2θ   
> ⇒ LHS = (1 + cot2θ - 1) tan2θ    
> ⇒ LHS = cot2θ . tan2θ    
> ⇒ LHS = (1/tan2θ) . tan2θ  
> ⇒ LHS = 1 = RHS
> 
> ∴ LHS = RHS [Hence Proved]

****Example 4: Prove that (sec********4********θ - sec********2********θ) = (tan********2********θ + tan********4********θ)****

****Solution:****

> We have:
> 
> LHS = (sec4θ - sec2θ)  
> ⇒ LHS = sec2θ(sec2θ - 1)  
> ⇒ LHS = (1 + tan2θ) (1 + tan2θ - 1)  
> ⇒ LHS = (1 + tan2θ) tan2θ  
> ⇒ LHS = (tan2θ + tan4θ) = RHS      
> 
> ∴ LHS = RHS [Hence Proved]

****Example 5: Prove that √(sec********2********θ + cosec********2********θ) = (tanθ + cotθ)**** 

****Solution:****

> We have:
> 
> LHS = √(sec2θ + cosec2θ ) = √((1 + tan2θ) + (1 + cot2θ))  
> ⇒ LHS = √(tan2θ + cot2θ + 2)  
> ⇒ LHS = √(tan2θ + cot2θ + 2tanθ.cotθ )         (tanθ . cotθ = 1)  
> ⇒ LHS = √(tanθ + cotθ)2  
> ⇒ LHS = tanθ + cotθ = RHS
> 
> ∴ LHS = RHS [Hence Proved]

## Practice Questions on Trigonometric Identities

Here are some practice questions to help you master key trigonometric identities and improve your problem-solving skills.

![PRACTICE-QUESTION](https://media.geeksforgeeks.org/wp-content/uploads/20250407122822446261/PRACTICE-QUESTION.png)

practice Questions Trigonometric Identities

> ****Practice Quiz -**** [****Trignometry Quiz****](https://www.geeksforgeeks.org/quizzes/trigonometry-quiz-questions-with-solutions/)

Suggested Quiz

10 Questions

If sec⁡ A + tan ⁡A = x, then the value of sec⁡ A − tan ⁡A is

- A
    
    1/x
    
- B
    
    x
    
- C
    
    1/x2
    
- D
    
    x2
    

If sin⁡ θ + cos ⁡θ = √2​, then sin⁡3θ + cos⁡3θ is equal to:

- A
    
    3√2/2
    
- B
    
    √2
    
- C
    
    √2/2
    
- D
    
    2
    

Find the value of \sin^6 \theta + \cos^6 \theta in terms of cos⁡2θ.

- A
    
    \frac{1}{4}(1 - 3\sin^2 2\theta)
    
- B
    
    \frac{1}{4}(1 - 3\cos^2 2\theta)
    
- C
    
    1 + \frac{3}{4} \cos^2 2\theta
    
- D
    
    1 - \frac{3}{8} \cos^2 2\theta
    

Value of tan 75°

- A
    
    2 + √3​
    
      
    
- B
    
    1 + √3​
    
- C
    
    2 - √3​
    
- D
    
    1 - √3​
    

\frac{\sin ^2 A-\sin ^2 B}{\sin A \cos A-\sin B \cos B}=

- A
    
    tan (A - B)
    
- B
    
    tan (A + B)
    
- C
    
    cot (A - B)
    
- D
    
    cot (A + B)
    

The sum of the solutions x∈R of the equation [Tex] \frac{3\cos⁡ 2x + \cos^3 ⁡2x}{\cos 6⁡x −\sin 6⁡x} [/Tex]= x3 − x2 + 6 is: [JEE Main 2024 29 January Evening Shift]

- A
    
    3
    
- B
    
    1
    
- C
    
    0
    
- D
    
    -1
    

If \sin A+\sin B=C, \cos A+\cos B=D, then the value of \sin (A+B)=

  

  

- A
    
    CD
    
- B
    
    [Tex]\frac{C D}{C^2+D^2}[/Tex]
    
- C
    
    [Tex]\frac{C^2+D^2}{2 C D}[/Tex]
    
- D
    
    [Tex]\frac{2 C D}{C^2+D^2}[/Tex]
    

If x + 1/x = 2 cos θ, then x3 + 1/x3 =

- A
    
    cos 3θ
    
- B
    
    2 cos 3θ
    
- C
    
    (1/2) cos 3θ
    
- D
    
    (1/3) cos 3θ
    

If y=(1 + tan A)(1 - tan B) where A - B = π/4, then (y+1)y+1 is equal to

  

- A
    
    9
    
- B
    
    4
    
- C
    
    27
    
- D
    
    81
    

If [Tex]a \sin ^2 x+b \cos ^2 x=c[/Tex], [Tex]b \sin ^2 y+a \cos ^2 y=d[/Tex] and [Tex]a \tan x=b \tan y[/Tex], then [Tex]\frac{a^2}{b^2}[/Tex], is equal to

- A
    
    [Tex]\frac{(b-c)(d-b)}{(a-d)(c-a)}[/Tex]
    
- B
    
    [Tex]\frac{(a-d)(c-a)}{(b-c)(d-b)}[/Tex]
    
- C
    
    [Tex]\frac{(d-a)(c-a)}{(b-c)(d-b)}[/Tex]
    
- D
    
    [Tex]\frac{(b-c)(b-d)}{(a-c)(a-d)}[/Tex]
    

![](https://media.geeksforgeeks.org/auth-dashboard-uploads/sucess-img.png)

Quiz Completed Successfully

Your Score :  2/10

Accuracy : 0%

View Explanation

1/10< Previous Next >


- [****Trigonometric Identities****](https://www.geeksforgeeks.org/maths/trigonometric-identities/)
- [****Real-life Application of trigonometry****](https://www.geeksforgeeks.org/maths/applications-of-trigonometric-functions/)
- [****Sin and Cos Formulas****](https://www.geeksforgeeks.org/maths/sin-cos-formulas-in-trigonometry-with-examples/)


- [Trigonometric Ratios](https://www.geeksforgeeks.org/maths/trigonometric-ratios/)
- [Trigonometry Table](https://www.geeksforgeeks.org/maths/trigonometry-table/)
- [Trigonometry Formulas](https://www.geeksforgeeks.org/maths/trigonometry-formulas/)
- [Trigonometric Functions](https://www.geeksforgeeks.org/maths/trigonometric-functions/)
- [Domain and Range of Trigonometric Functions](https://www.geeksforgeeks.org/maths/domain-and-range-of-trigonometric-functions/)
- [Graph of Trigonometric Functions](https://www.geeksforgeeks.org/maths/trigonometric-graph/)
- [Application of Trigonometry in Real Life](https://www.geeksforgeeks.org/maths/applications-of-trigonometry/)
- [Height and Distance](https://www.geeksforgeeks.org/maths/height-and-distance/)
- [Trigonometric Equations](https://www.geeksforgeeks.org/maths/trigonometric-equations/)
- [Trigonometric Symbols](https://www.geeksforgeeks.org/maths/trigonometric-symbols/)

## Trigonometry for Aptitude

This section focuses on trigonometry concepts and aptitude quizzes, including height and distance problems, equations, identities, and non-right angle applications.

- [Trigonometry - Quiz](https://www.geeksforgeeks.org/quizzes/trigonometry-quiz-questions-with-solutions/)
- [Height and Distances - Aptitude Questions and Answers](https://www.geeksforgeeks.org/aptitude/height-and-distance-questions-aptitude/)
- [Height and Distances - Quiz for Aptitude](https://www.geeksforgeeks.org/quizzes/trigonometry-height-and-distances-gq/)
- [Trigonometric Equations and Identities - Quiz](https://www.geeksforgeeks.org/quizzes/trigonometric-equations-and-identities/)
- [Trigonometry for Non-Right Angles - Quiz](https://www.geeksforgeeks.org/quizzes/non-right-triangles-trigonometry/)

## Trigonometry Practice Questions

This section offers a range of trigonometry practice questions, from basic to advanced, covering equations, identities, and ratios to help strengthen your understanding and skills.

- [Trigonometry Practice Questions Easy](https://www.geeksforgeeks.org/maths/trigonometry-questions-easy/)
- [Trigonometry Practice Questions Medium  
    ](https://www.geeksforgeeks.org/maths/trigonometry-practice-questions-medium/)
- [Trigonometry Practice Questions Hard](https://www.geeksforgeeks.org/maths/trigonometry-practice-questions-hard/)
- [Trigonometric Equations Practice Questions](https://www.geeksforgeeks.org/maths/trigonometric-equations-practice-questions/)
- [Trigonometric Identities Practice Problems](https://www.geeksforgeeks.org/maths/trigonometric-identities-practice-problems/)
- [Trigonometric Ratios Practice Questions](https://www.geeksforgeeks.org/maths/trigonometric-ratios-practice-questions/)

## Trigonometry for Programming

This section covers the implementation of trigonometric functions in different programming languages like C++, Java, Python, MATLAB, and LaTeX, helping you apply trigonometry in coding environments.

- [C++ Program to Illustrate Trigonometric Functions](https://www.geeksforgeeks.org/dsa/c-program-to-illustrate-trigonometric-functions/)
- [Trigonometric Functions in Java](https://www.geeksforgeeks.org/java/trigonometric-functions-in-java-with-examples/)
- [Trigonometric and Angular Functions in Python](https://www.geeksforgeeks.org/python/mathematical-functions-in-python-set-3-trigonometric-and-angular-functions/)
- [Trigonometric Functions in MATLAB](https://www.geeksforgeeks.org/engineering-mathematics/trigonometric-functions-in-matlab/)
- [Trigonometric Functions in LaTex](https://www.geeksforgeeks.org/engineering-mathematics/trigonometric-functions-in-latex/)


Trigonometric functions are the basic functions used in [trigonometry](https://www.geeksforgeeks.org/maths/math-trigonometry/) and they are used for solving various types of problems in physics, Astronomy, Probability, and other branches of science. There are six basic trigonometric functions used in Trigonometry which are:

- [Sine Function](https://www.geeksforgeeks.org/maths/sine-function/) (sin x)
- [Cosine Function](https://www.geeksforgeeks.org/maths/cosine-function/) (cos x)
- [Secant Function](https://www.geeksforgeeks.org/maths/secant-formula-concept-formulae-solved-examples/) (tan x)
- [Cosecant Function](https://www.geeksforgeeks.org/maths/cosecant-formula/) (cosec x)
- [Tangent Function](https://www.geeksforgeeks.org/maths/tangent-function/) (tan x)
- [Cotangent Function](https://www.geeksforgeeks.org/maths/cotangent-formula/) (cot x)

## Six Trigonometric Functions

The image added below shows a right-angle triangle PQR.

![Right-Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230828140037/Right-Triangle-Formula-1-(2)-(1).png)

Then the six basic trigonometric functions formulas for this right angle triangle are,

|Function|Sides|Description|Relation|
|---|---|---|---|
|sin θ|PQ/PR|Perpendicular/Hypotenuse|sin θ = 1/csc θ|
|cos θ|QR/PR|Base/Hypotenuse|cos θ = 1/sec θ|
|tan θ|PQ/QR|Perpendicular/Base|tan θ = 1/cot θ|
|sec θ|PR/PQ|Hypotenuse/Base|sec θ = 1/cos θ|
|cosec θ|PR/QR|Hypotenuse/Perpendicular|cosec θ = 1/sin θ|
|cot θ|QR/PQ|Base/Perpendicular|cot θ = 1/tan θ|

> ****Read More:**** [Trigonometric function Ratios](https://www.geeksforgeeks.org/maths/trigonometric-ratios/)

### Values of Trigonometric Functions

The value of trigonometric functions can easily be given using the trigonometry table. These values of the trigonometric functions are very useful in solving various trigonometric problems. The required [trigonometry table](https://www.geeksforgeeks.org/maths/trigonometry-table/) is added below:

![Trogonometry-2](https://media.geeksforgeeks.org/wp-content/uploads/20231121235818/Trogonometry-2.png)

The table added above shows all the values of the important angles from 0 to 180 degrees for all the trigonometric functions.

### Trigonometric Functions in Four(4) Quadrants

The trigonometric functions are the periodic functions and their values repeat after a certain interval. Also, not all the trigonometric functions are positive in all the [quadrants](https://www.geeksforgeeks.org/maths/quadrant/).

An image explaining the same is added below:

![Trigonometric Functions in Quadrant](https://media.geeksforgeeks.org/wp-content/uploads/20230828170310/image-(7).png)

We divide the cartesian space into four quadrants namely, I, II, III, and IV quadrants, and the value of the trigonometric functions whether they are positive or negative in each quadrant is given as,

- ****I Quadrant:**** All Positive
- ****II Quadrant:**** sin θ and cosec θ Positive
- ****III Quadrant:**** tan θ and cot θ Positive
- ****IV Quadrant:**** cos θ and sec θ Positive

## Trigonometric Functions Graph

Trigonometric functions graphs plot the value of the trigonometric functions for different values of the angle(θ). For some the trigonometric functions are bounded as,

- Trigonometric functions sin θ and cos θ are bounded between - 1 and 1 and their graphs oscillate between -1 and 1 on the y-axis.
- Graph of the trigonometric function tan θ, and cot θ has a range from negative infinity to positive infinity.
- Graph of the trigonometric function sec θ, and cosec θ has a range from negative infinity to positive infinity excluding (-1, 1).

> ****Read More:**** [Graph of Trigonometric Functions](https://www.geeksforgeeks.org/maths/trigonometric-graph/)

## Domain and Range of Trigonometric Functions

Suppose we have a trigonometric function f(x) = sin x, then the domain of the function f(x) is all the values of x that the function f(x) can take, and the domain is all possible outcomes of the f(x). The domain and range of all the six trigonometric functions are:

|Trigonometric Function|Domain|Range|
|---|---|---|
|sin x|R|[-1, +1]|
|cos x|R|[-1, +1]|
|tan x|R - (2n + 1)π/2|R|
|cot x|R - nπ|R|
|sec x|R - (2n + 1)π/2|(-∞, -1] U [+1, +∞)|
|cosec x|R - nπ|(-∞, -1] U [+1, +∞)|

> ****Read in Detail-**** [****Domain and Range of Trigonometric Functions****](https://www.geeksforgeeks.org/maths/domain-and-range-of-trigonometric-functions/)****.****

## Properties of Trigonometric Functions

Some of the common properties of trigonometric functions are discussed below:

****Period**** refers to the length of one complete cycle of a trigonometric function, after which the function repeats.

- Sine (sin), Cosine (cos), Secant (sec), Cosecant (csc): Period = 2π
- Tangent (tan), Cotangent (cot): Period = π

****Symmetry**** refers to the property that describes how the function behaves under reflection, translation, or rotation.

- [****Even Functions****](https://www.geeksforgeeks.org/maths/even-function/): f(−θ) = f(θ) (Cosine and Secant).
- [****Odd Functions****](https://www.geeksforgeeks.org/maths/odd-function-definition-properties-and-examples/): f(−θ) = −f(θ) (Sine, Tangent, Cosecant, Cotangent).

### ****Derivatives of Trigonometric Functions****

[Differentiation of trigonometric function](https://www.geeksforgeeks.org/maths/differentiation-of-trigonometric-functions/) can be easily found and the slope of that curve for that specific value of the trigonometric functions. The differentiation of all six trigonometric functions is added below:

- d/dx (sin x) = cos x
- d/dx (cos x) = -sin x
- d/dx (tan x) = sec2x
- d/dx (cot x) = -cosec2x
- d/dx (sec x) = sec x tan x
- d/dx (cosec x) = -cosec x cot x

### ****Integration of Trigonometric Functions****

As the integration of any curve gives the area under the curve, the [integration of the trigonometric function](https://www.geeksforgeeks.org/maths/integration-of-trigonometric-functions/) also gives the area under the trigonometric function. The integration of various trigonometric functions is added below.

- ∫ cos x dx = sin x + C
- ∫ sin x dx = -cos x + C
- ∫ tan x dx = log|sec x| + C
- ∫ cot x dx = log|sin x| + C
- ∫ sec x dx = log|sec x + tan x| + C
- ∫ cosec x dx = log|cosec x - cot x| + C

Some other important trigonometric integrals are:

- ∫ sec2x dx = tan x + C
- ∫ cosec2x dx = -cot x + C
- ∫ sec x tan x dx = sec x + C
- ∫ cosec x cot x dx = -cosec x + C

****Related Reads:****,

> - [Application of Trigonometry in Real Life](https://www.geeksforgeeks.org/maths/applications-of-trigonometry/)
> - [Trigonometric Equations](https://www.geeksforgeeks.org/maths/trigonometric-equations/)
> - [Trigonometric Symbols](https://www.geeksforgeeks.org/maths/trigonometric-symbols/)
> 
> Various functions used in trigonometry are called [trigonometry functions](https://www.geeksforgeeks.org/maths/trigonometric-functions/) they define the relationships between angles and sides of the triangle. The six basic [trigonometric formulas](https://www.geeksforgeeks.org/maths/trigonometry-formulas/) are sin, cosine, tan, cosec, sec, and cot.

## What are Sum and Difference Formulas?

Sum and Difference formulas are used to calculate the trigonometric function for those [angles](https://www.geeksforgeeks.org/maths/angle-definition/#:~:text=) where standard angle can't be used. We have six main sum and difference formulas which we mainly used in trigonometry.

The six main trigonometric sum and difference formulae are given in the image below:

![Sum-and-Difference-Formulas-copy](https://media.geeksforgeeks.org/wp-content/uploads/20240607144707/Sum-and-Difference-Formulas-copy.webp)

Trigonometric Sum and Difference Formulas

Table of Content

- [Proof of Sum and Difference Identities](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/#proof-of-sum-and-difference-identities)
- [Sum and Difference Formulas for Cosine](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/#sum-and-difference-formulas-for-cosine)
- [Sum and Difference Formulas for Sine](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/#sum-and-difference-formulas-for-sine)
- [Sum and Difference Formulas for Tangent](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/#sum-and-difference-formulas-for-tangent)
- [Sum and Difference Formulae Table](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/#sum-and-difference-formulae-table)
- [How to Apply Sum and Difference Formulas](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/#how-to-apply-sum-and-difference-formulas)
- [Solved Examples on Sum and Difference Formulas](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/#solved-examples-on-sum-and-difference-formulas)

### Sum and Difference Identities

****Sine Formulas:****

- sin (A + B) = sin A cos B + cos A sin B
- sin (A – B) = sin A cos B – cos A sin B

****Cos Formulas:****

- cos (A + B) = cos A cos B – sin A sin B
- cos (A – B) = cos A cos B + sin A sin B

****Tan Formulas:****

- tan (A + B) = (tan A + tan B)/(1 – tan A tan B)
- tan (A – B) = (tan A – tan B)/(1 + tan A tan B)

## ****Proof of Sum and Difference Identities****

To demonstrate, the trigonometric sum and difference formulas let us consider a unit circle, with coordinates given as (cos θ, sin θ).

- Consider points A and B, which form angles of α and β with the positive X-axis, respectively.
- The coordinates of A and B are (cos α, sin α) and (cos β, sin β), respectively.

We can observe that the angle AOB is equal to (α - β). Now, consider another two points P and Q on the unit circle such that Q is a point on the X-axis with coordinates (1,0) and angle POQ is equal to (α - β), and thus the coordinates of the point P are (cos (α - β), sin (α - β)). 

![Sum and Difference Identities](https://media.geeksforgeeks.org/wp-content/uploads/20230210172749/Sum-and-Difference-Identites.png)

Now, OA = OP, and OB = OQ as they are the radii of the same unit circle, and also the measure of one of the included angles of both triangles is (α - β).

Hence, by the side-angle-side congruence, triangles AOB and triangle POQ are congruent.

We know that the corresponding parts of congruent triangles are congruent, hence AB = PQ.

So, AB = PQ.

Using the distance formula between two points we get,

****d********AB**** ****= √[(cos α - cos β)********2**** ****+ (sin α - sin β)********2********]****

= √[cos2 α - 2 cos α cos β + cos2 β + sin2 α - 2 sin α sin β + sin2 β]    {Since, (a - b)2 = a2 - 2ab + b2)}

= √[(cos2 α+ sin2 α) + (cos2 β+ sin2 β) - 2(cos α cos β + sin α sin β)]

= √[1 + 1 - 2(cos α  cos β + sin α  sin β)]         {Since, sin2 x + cos2 x = 1}

> = √[2 - 2(cos α cos β+ sin α sin β)].......(1)

****d********PQ**** ****= √[(cos (α - β) - 1)********2**** ****+ (sin (α - β) - 0)********2********]**** 

= √[cos2 (α - β) - 2 cos (α - β) + 1 + sin2 (α - β)]      {Since, (a - b)2 = a2 - 2ab + b2)}

= √[(cos2 (α - β) + sin2 (α - β)) + 1 - 2 cos (α - β)]

= √[1 + 1 - 2 cos (α - β)]            {Since, sin2 x + cos2 x = 1}

> = √[2 - 2 cos (α - β)]......(2)

Since AB = PQ, equate both equations (1) and (2).

√[2 - 2(cos α cos β+ sin α sin β)] = √[2 - 2 cos (α - β)] 

By squaring on both sides, we get,

> 2 - 2(cos α cos β+ sin α sin β) = 2 - 2 cos (α - β)......(3)

## Sum and Difference Formulas for Cosine

### Cos (α - β) formula

> from eq (3)
> 
> 2 (1 - cos α cos β - sin α sin β) = 2 (1 - cos (α - β))  
> 1 - cos α cos β - sin α sin β = 1 - cos (α - β)
> 
> ****cos (α - β) = cos α cos β + sin α sin β****

### Cos (α + β) formula

> To derive the sum formula of the cosine function substitute (-β) in the place of β in the difference of the cosine function.
> 
> Hence, cos (α + β) = cos (α - (β))  
>                  = cos α cos (-β) + sin α sin (-β)    ****{Since, cos (α - β) = cos α cos β + sin α sin β}****  
>                   = cos α cos β - sin α sin β            ****{Since, cos (-θ) = cos θ, sin (-θ) = - sin θ}****
> 
> ****cos (α + β) = cos α cos β - sin α sin β****

## Sum and Difference Formulas for Sine

### Sin (α - β) formula

> We know that, ****sin (90° - θ) = cos θ and cos (90° - θ) = sin θ****. 
> 
> So, sin (α - β) = cos (90° - (α - β))  
>                  = cos (90° - α + β)  
>                  = cos [(90° - α) + β]  
>                  = cos (90° - α) cos β - sin (90° - α) sin β      ****{Since,  cos (α + β) = cos α cos β - sin α sin β}****
> 
> ****sin (α - β) = sin α cos β - cos α sin β****

### Sin (α + β) formula

> We know that, ****sin (90° - θ) = cos θ and cos (90° - θ) = sin θ****.
> 
> So, sin (α + β) = cos (90° - (α + β))  
>                  = cos (90° - α - β)  
>                  = cos [(90° - α) - β]  
>                  = cos (90° - α) cos β + sin (90° - α) sin β    ****{Since, cos (α - β) = cos α cos β + sin α sin β}****
> 
> ****sin (α + β) = sin α cos β + cos α sin β****

## Sum and Difference Formulas for Tangent

### Tan (α - β) formula

> We know that, tan θ = sin θ/cos θ
> 
> So, tan (α - β) = sin (α - β)/cos (α - β)  
>                  = (sin α cos β - cos α sin β)/(cos α cos β + sin α sin β)
> 
> Now, divide the numerator and denominator with cos α cos β  
>                = [(sin α cos β - cos α sin β)cos α cos β ]/[(cos α cos β + sin α sin β)/(cos α cos β)  
>                  = (sin α/cos α - sin β/cos β)/(1 + (sin α/cos α)×(sin β/cos β))  
>                  = (tan α - tan β)/(1 + tan α tan β)  
> ****tan (α - β) = (tan α - tan β)/(1 + tan α tan β)****

### Tan (α + β) formula

> To derive the tan (α + β) formula substitute (-β) in the place of β in the tan (α - β) formula.
> 
> Hence, we get, 
> 
> tan (α + β) = tan(α - (-β))
> 
>                   = (tan α - tan (-β))/(1 + tan α tan (-β))            {Since, tan (α - β) = (tan α - tan β)/(1 + tan α tan β)}  
>                   = (tan α + tan β)/(1 - tan α tan β)                   {Since, tan (-θ) = - tan θ}  
> ****tan (α + β) = (tan α + tan β)/(1 - tan α tan β)****

## Sum and Difference Formulae Table

In the previous section, we derived the formulas of all the sum and difference identities of the [trigonometric functions](https://www.geeksforgeeks.org/maths/trigonometric-functions/) sine, cosine, and tangent. Now, let us summarize these formulas in the table below for a quick revision.

||Sum Formulae|Difference Formulae|
|---|---|---|
|Sine function|sin (α + β) = sin α cos β + cos α sin β|sin (α - β) = sin α cos β - cos α sin β|
|Cosine function|cos (α + β) = cos α cos β - sin α sin β|cos (α - β) = cos α cos β + sin α sin β|
|Tangent function|tan (α + β) = (tan α + tan β)/(1 - tan α tan β)|tan (α - β) = (tan α - tan β)/(1 + tan α tan β)|

## How to Apply Sum and Difference Formulas

[Sum](https://www.geeksforgeeks.org/maths/sum/) and Difference Formulas of [trigonometry](https://www.geeksforgeeks.org/maths/math-trigonometry/) are used to solve various trigonometry problems and find the values of trigonometric functions without standard values. To Apply Sum and Difference Formulas study the following example,

****Example: Find the value of sin 15°****

****Solution:****

> ****Step 1:**** Write the given function in the sum and difference of the standard function,  
> sin 15° = sin (45 -30)°
> 
> ****Step 2:**** Use the required Sum and Difference Formulas, here we use, sin (α - β) = sin α cos β - cos α sin β  
> sin (45 -30)° = sin 45° cos 30° - cos 45° sin 30°
> 
> ****Step 3:**** Substitute the value of these standard trigonometric functions using the [trigonometric table](https://www.geeksforgeeks.org/maths/trigonometry-table/).  
> sin (45 -30)° = 1/√2 × √3/2 - 1/√2 × 1/2
> 
> ****Step 4:**** Simplify the value obtained in the above step.  
> sin (45 -30)° = 1/√2 × √3/2 - 1/√2 × 1/2  
>                     = (√3 -1)/ 2√2 
> 
> ****sin 15° = (√3 -1)√2 / 4****

****Read More:****

> - [Pythagoras Theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/)
> - [Trigonometric Ratios](https://www.geeksforgeeks.org/maths/trigonometric-ratios/)
> - [Heights and Distances](https://www.geeksforgeeks.org/maths/height-and-distance/)

## ****Solved Examples of Sum and Difference Formulas****

****Example 1: Prove the triple angle formulae of sine and cosine functions using the sum and difference formulae.****

-  ****sin 3A = 3 sin A - 4 sin********3********A****
- ****cos 3A = 4 cos********3**** ****A - 3 cos A****

****Solution:****

****To Prove:**** sin 3A = 3 sin A - 4 sin3A

> sin 3A = sin (2A + A)     ****[sin (A + B) = sin A cos B + cos A sin B]****
> 
> sin (2A + A) = sin 2A cos A + cos 2A sin A
> 
> We know that,
> 
> ****sin 2A = 2 sin A cos A, and cos 2A = 1 - 2sin********2**** ****A, and cos********2**** ****A = 1 - sin********2**** ****A****
> 
> sin (2A + A) = (2 sin A cos A) cos A + (1 - 2sin2 A)sin A  
>                     = 2 sin A cos2 A + sin A - 2 sin3 A   
>                     = 2 sin A (1 - sin2 A) + sin A - 2 sin3 A  
>                     = 2 sin A - 2sin3 A + sin A - 2 sin3 A  
>                     = 3 sin A - 4 sin3 A
> 
> Thus, sin 3A = 3 sin A - 4 sin3 A  (proved)

****To Prove: cos 3A = 4 cos********3**** ****A - 3 cos A****

> cos 3A = cos (2A + A)   ****[cos (A + B) = cos A cos B - sin A sin B]****
> 
> So, cos (2A + A) = cos 2A cos A - sin 2A sin A
> 
> We know that, 
> 
> ****sin 2A = 2sin A cos A and cos 2A = 2cos********2**** ****A - 1, and sin********2**** ****A = 1- cos********2**** ****A****
> 
> cos (2A + A) = (2 cos2 A - 1) cos A - (2 sin A cos A) sin A  
>                     = 2 cos3 A - cos A - 2 sin2 A cos A  
>                     = 2 cos3 A - cos A - 2 (1- cos2 A) cos A  
>                     = 2 cos3 A - cos A - 2 cos A + 2 cos3 A   
>                     = 4 cos3 A - 3 cos A
> 
> Thus, cos 3A = 4 cos3 A - 3 cos A  (proved)

****Example 2: Find the value of cos 75° using the sum and difference formulae.****

****Solution:****

> We can write 75° as the sum of 45° and 30°
> 
> cos 75° = cos (45° + 30°)
> 
>             = cos 45° cos 30° - sin 45° sin 30°             ****{Since, cos (A + B) = cos A cos B - sin A sin B}****  
>             = (1/√2) (√3/2) - (1/√2)(1/2)                    ****{Since, cos 45° = sin 45° = (1√2) , cos 30° = √3/2, sin 30° = 1/2}****  
>             = (√3 -1)/2√2
> 
> Hence, cos 75° = (√3 - 1)/2√2

****Example 3: Find the value of tan 105° using the sum and difference formulae.****

****Solution:****

> We can write 105° as the sum of 60° and 45°.
> 
> tan 105° = tan (60° + 45°)
> 
>               = (tan 60° + tan 45°)/(1 - tan 60° tan 45°)   ****{Since, tan (A + B) = (tan A + tan B)****  
>               = (√3 + 1)/(1 - (√3 × 1))                              ****{Since, tan 60° = √3, tan 45° = 1}****  
>               = (√3 + 1)/(1 - √3)
> 
> Rationalize the above expression with the conjugate of the denominator,
> 
>             = [3+11−3]×[1+31+3][1−3​3​+1​]×[1+3​1+3​​]
> 
>             = (√3 + 1)2/(1 - (√3)2)  
>             = (3 + 2√3 + 1)/(1 - 3)  
>             = (4 + 2√3)/(-2)  
>             = -2 - √3
> 
> Hence, tan 105° = -2 - √3.

****Example 4: Evaluate the value of sin 15° using the sum and difference formulae.****

****Solution:****

> We can write 15° as the difference between 45° and 30°
> 
> sin 15° = sin (45° - 30°)
> 
>             = sin 45° cos 30° - cos 45° sin 30°   ****{Since, sin (A - B) = sin A cos B - cos A sin B}****  
>             = (1/√2) (√3/2) - (1/√2)(1/2)          ****{Since, cos 45° = sin 45° = (1√2) , cos 30° = √3/2, sin 30° = 1/2}****  
>             = (√3 - 1)/2√2
> 
> Hence, sin 15° = (√3 - 1)/2√2

****Example 5: Prove that sin (π/4 - a) cos (π/4 - b) + cos (π/4 - a) sin (π/4 - b)  = cos (a + b).****

****Solution:****

> L.H.S = sin (π/4 - a) cos (π/4 - b) + cos (π/4 - a) sin (π/4 - b)   ****{sin (A + B) = sin A cos B + cos A sin B}****
> 
>          = sin [(π/4 - a) + (π/4 - b)]  
>          = sin [(π/2) - (a + b)]  
>           = cos (a + b)             ****{Since, sin (90° - θ) = cos θ}****  
>            = R. H. S    (proved)

Using angle [sum and difference identities](https://www.geeksforgeeks.org/maths/sum-and-difference-identities/), we get,

sin (α + γ) = sin α cos γ + sin γ cosα

⇒ c (sin α cos γ + sin γ cos α ) = b sin γ

⇒ c sin α = a sin γ

Dividing the whole equation by cos γ,

c (sin α + tan γ cos α) = b tan γ

⇒ c sin α /cos γ = a tan γ

⇒ c2sin2 α / cos2 γ = tan γ

From equation 1, we get,

c sin α / b – c cos α = tan γ

⇒ 1 + tan2 γ = 1/cos2 γ

⇒ c2 sin2 α (1+ (c2 sin2α / (b – c cos α )2)) = a2 (c2 sin2α / (b – c cos α )2)

Multiplying the equation by (b – c cos α )2 and arranging it,

a2 = b2 + c2 – 2bc cos α.

Hence, using algebraic manipulation cosine rule is proved.

## Properties of Cosine Rule

The properties of cosine rule are listed below

- To determine the side lengths of triangle ABC, we can express it as a2 = b2 + c2 – 2bc cos α, b2 = a2 + c2 – 2ac cos β and c2 = b2 + a2 – 2ba cos γ.
- To calculate the angles of triangle ABC, the Cosine Rule is expressed in the following manner: cos α = [b2 + c2 – a2]/2bc, cos β = [a2 + c2 – b2]/2ac and cos γ = [b2 + a2 – c2]/2ab.
- Sine Rule is expressed as a/sin α = b/sin β = c/sin γ.
- Cosine Rule is also known as the Law of Cosines.
- Cosine Rule is a formula which helps to calculate the sides and angles of a triangle.
- [Pythagoras Theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/) is an application of the Cosine Rule which only holds true for [right-angle triangles](https://www.geeksforgeeks.org/maths/right-angled-triangle/).
- The Cosine Rule can be applied or utilized in any triangle.

### Where to use Cosine Rule?

The Cosine Rule is useful for finding:

- The third side of a triangle by using the lengths of two known sides and the angle formed between them.
- The angles of a triangle by using all three sides of the triangle.

## Examples of the Cosine Rule

We can use cosine rule for:

- Finding Sides
- Finding Angles

Let's discuss the method for finding sides and angles in detail as follows:

### Finding Missing Length Using the Cosine Rule

Cosine Rule can be used to calculate the unknown parameters of a triangle when all known elements are given. Let us understand the process of finding out the missing side or angle of a triangle using the Cosine Rule.

> ****Step 1:**** Note down the given parameters like side lengths and measure of angles for the triangle and identify the element to be calculated.
> 
> ****Step 2:**** Apply the cosine rule formulas,
> 
> - a2 = b2 + c2 – 2bc cos α
> - b2 = a2 + c2 – 2ac cos β
> - c2 = a2 + b2 – 2ab cos γ
> 
> where, α, β, and γ are the angle of a triangle, and their opposite sides are represented as a, b, and c respectively.
> 
> ****Step 3:**** Represent the result with suitable units.

### Cosine Rule To Find Angles

Cosine Rule can be used to find unknown angles in a Triangle using the formula given below:

- cos A = (b2 + c2 − a2)/2bc
- cos B = (a2 + c2 − b2)/2ac
- cos C = (a2 + b2 − c2)/2ab

Example 4 in Solved Examples deals with Cosine Rule to Find Angles

## Sine and Cosine Rule

[Sine Rule and Cosine Rule](https://www.geeksforgeeks.org/maths/law-of-sine-and-cosine-formula/) are important rules in Trigonometry to establish the relation between angles and sides of a triangle. A detailed comparioson between Sin Rule and Cosine Rule is discussed in the table below:

|[Sine Rule](https://www.geeksforgeeks.org/maths/sine-rule/)|Cosine Rule|
|---|---|
|Sine Rule states that "the ratio of side to the sine of the angle opposite to it always remains constant"|Cosine Rule states that “the square of one side of a triangle, equals the sum of the squares of the other two sides, subtracted by twice the product of those two sides and the cosine of the angle between them.”|
|****Sine Rule Formula:****<br><br>- a/Sin A = b/Sin B = c/Sin C|****Cosine Rule Formula:****<br><br>- a2 = b2 + c2 – 2bc cos α<br>- b2 = a2 + c2 – 2ac cos β<br>- c2 = a2 + b2 – 2ab cos γ|

****Also, Check****

- [****Trigonometry****](https://www.geeksforgeeks.org/maths/math-trigonometry/)
- [****Trigonometric Formulas****](https://www.geeksforgeeks.org/maths/trigonometry-formulas/)
- [****Cosine Formulas****](https://www.geeksforgeeks.org/maths/what-are-cosine-formulas/)

## Solved Examples on Cosine Rule

****Example 1. Determine the angle of triangle ABC if AB = 42cm, BC = 37cm and AC = 26cm?****

****Solution:****

> As per the question we have following given data:
> 
> a = 42cm b = 37cm and c = 26cm
> 
> Formula of Cosine Rule: a2 = b2 + c2 − 2bc cos α
> 
> So, 422 = 372 + 262 − 2(37)(26) cos α
> 
> cos α = 372 + 262 − 422 /(2)(37)(26) 985
> 
> After solving the cos α we get the value of α as
> 
> cos α = 1071/2184
> 
> ⇒ cos α = 0.4904
> 
> Thus, α = cos −1 0.4904 = 60.63°

****Example 2. Two sides of a triangle measure 70 in and 50 in with the angle between them measuring 49º. Find the missing side.****

****Solution:****

> Put the value 72 inch for b, 50inch for c and 49º for α.
> 
> Using the Cosine Rule formula,
> 
> a2 = b2 + c2 - 2bccos α
> 
> ⇒ a2 = (70)2 + (50)2 - 2(72)(50)cos49º
> 
> ⇒ a2 = 4900 + 2500 - (7200)(0.656)
> 
> ⇒ a2 = 4900+ 2500 - 4723.2
> 
> ⇒ a2 = 2676.8
> 
> ⇒ a ≈ 51.73
> 
> So, the missing length of the side is 51.73 inches.

****Example 3. How long is side "c", when we know the angle C = 37°, and sides a = 9 and b = 11.****

****Solution:****

> The Cosine Rule is c2 = a2 + b2 − 2ab cos(C)
> 
> Put the given values we know: c2 = 92 + 112 − 2 × 9 × 11 × cos(37º)
> 
> c2 = 81 + 121 − 198 × 0.798
> 
> ⇒ c2 = 43.99
> 
> ⇒ c = √43.99 = 6.63.

****Example 4: Find Angle "C" using the Cosines Rule (angle version). If in this triangle we know the three sides: a = 8, b = 6 and c = 7.****

****Solution:****

> Use The Cosines Rule to find angle C :
> 
> cos C= (a2 + b2 − c2)/2ab
> 
> ⇒ cos C = (82 + 62 − 72)/2×8×6
> 
> ⇒ cos C = (64 + 36 − 49)/96
> 
> ⇒ cos C = 51/96
> 
> ⇒ cos C = 0.53125
> 
> ⇒ C= cos−1(0.53125)
> 
> ⇒ cos C = 57.9°

## Practice Question on Cosine Rule

****Q1.**** Determine the angle of triangle ABC if AB = 32cm, BC = 30cm and AC = 24cm.

****Q2.**** Two sides of a triangle measure 62 in and 40 in with the angle between them measuring 50º. Find the missing side.

****Q3.**** How long is side "c", when we know the angle C = 47º, and sides a = 11 and b = 15.

****Q4.**** Find Angle "C" using the Cosines Rule (angle version). If in this triangle we know the three sides: a = 9, b = 6 and c = 8.

****Q5.**** Express sin 12θ + sin 4θ as the product of sines and cosines.

Heron's Formula can also be easily solved using the [Cosine Rule](https://www.geeksforgeeks.org/maths/cosine-rule/). Now for any triangle ABC if the sides of the triangle are a, b, and c and their opposite angles are, α, β, and γ.

The law of cosine states, cos γ = (a2 + b2 - c2)/2ab

Using [Trigonometric identites](https://www.geeksforgeeks.org/maths/trigonometric-identities/)

cos2 γ + sin2 γ = 1  
⇒ sin γ = √(1 - cos2 γ)  
⇒ sin γ = √[1 - {(a2 + b2 - c2)/2ab}2]  
⇒ sin γ = √[(4a2b2 - (a2 + b2 + c2)2]/2ab

If the base of the triangle is ****a**** then its altitude is ****b sin γ****

****Area of Triangle = 1/2 base × height****  
⇒ Area of Triangle = 1/2 × a × b sin γ  
⇒ Area of Triangle = 1/2 ab × √[(4a2b2 - (a2 + b2 + c2)2]/2ab  
⇒ Area of Triangle = 1/4 √[c2 - (a - b)2][(a + b)2 - c2]  
⇒ Area of Triangle = √(b + c - a)(a + c - b)(a + b - c)(a + b + c)/16

> ****Area of Triangle = √s(s - a)(s - a)(s - b)****
> 
> Where ****s = (a + b + c)/2**** is the semi perimeter.

## Heron’s Formula for Equilateral Triangle

For an [equilateral triangle](https://www.geeksforgeeks.org/maths/equilateral-triangle/), all sides are equal. Now, the semi-perimeter of the equilateral triangle is (s) = (a + a + a) / 2  
⇒ s = 3a / 2  
where a is the length of the side.

Now, using Heron’s Formula,  
Area of Equilateral Triangle = √(s(s – a)(s – a)(s – a), upon multiplication, the formula becomes:

> ****Area of Equilateral Triangle = √3 / 4 × a********2****

## Heron's Formula for Isosceles Triangle

[Isosceles Triangle](https://www.geeksforgeeks.org/maths/isosceles-triangle/) is a triangle that has two equal sides, their area can be easily calculated using Heron's Formula. For any isosceles triangle △ABC where sides AB = a, and BC = a are equal, and the third side is CA = b. The formula for its area is,

> ****Area of iscosceles triangle ABC(A) = √s(s - a)(s - a)(s - b)****

Where ****s**** is the semi-perimeter i.e., ****s = (a + a + c)/2****.  
⇒ s = (a + a + b)/2  
⇒ s = (2a + b)/2

Simplifying, A = √s(s - a)(s - a)(s - b)

****Substituting "s" = (2a + b)/2:****

⇒ s − a = (2a + b)/2 ​− a = b/2​  
⇒ s − b = (2a + b)/2 ​− b = (2a − b)/2​

Thus: a=(2a+b2)(b2)(b2)(2a−b2)a=(22a+b​)(2b​)(2b​)(22a−b​)​  
Simplyfy further: A=(2a+b)(2a−b)(b2)2A=2(2a+b)(2a−b)(b2)​​  
A = 1/4√(4a2 - b2)(b2)

****Final Formula:****

> A=b2a2−b24A=2b​a2−4b2​​

Note: Heron's formula for a ****Scalene triangle**** remains the same as the default formula, as it applies to all types of triangles, including scalene, isosceles, and equilateral, provided the lengths of the three sides are known.

## Heron's Formula for Area of Quadrilateral

Heron's formula is used to determine the formula for the area of the quadrilateral. We can divide the quadrilateral into two separate triangles using any one of its diagonals and then the area of the two separate triangles is calculated using Heron's Formula.

The area of the quad ABCD is calculated by dividing it into two triangles using its diagonal. Let's say we join the vertices A and C to form the diagonal AC then we divide it into two triangles △ABC and △ADC. If we take the length of sides of the quadrilateral as,

AB = b, BC = c, CD = d, and DA = a, and the length of diagonal AC is e then its area is calculated using,

![Heron-Formula-for-Quadrilateral](https://media.geeksforgeeks.org/wp-content/uploads/20241205165316716754/Heron-Formula-for-Quadrilateral.webp)

Heron's Formula for the Area of Quadrilateral

  

Area of quad ABCD = Area of △ABC + Area of △ADC...(i)

Area of triangle ABC

> ****Area of △ ABC = √(s********1********(s********1**** ****- b)(s********1**** ****- c)(s********1**** ****- e))****
> 
> Where ****s********1**** ****= (b + c + e)/2.****

Area of triangle ADC

> ****Area of △ ADC = √(s********2********(s********2**** ****- d)(s********2**** ****- a)(s********2**** ****- e))****
> 
> Where ****s********2**** ****= (d + a + e)/2.****

Thus,  
form eq(i)

> ****Area of quad ABCD = √(s********1********(s********1**** ****- b)(s********1**** ****- c)(s********1**** ****- e)) + √(s********2********(s********2**** ****- d)(s********2**** ****- a)(s********2**** ****- e))****
> 
> Where,
> 
> - ****s********1**** ****= (a + b + e)/2,**** and
> - ****s********2**** ****= (a + d + e)/2****

## Applications of Heron's Formula

Heron's formula has various applications and some of the important applications of Heron's Formula are,

- For finding the area of the triangle if the sides of the triangle are given
- For finding the area of the quadrilateral the length of all the sides and the length of the diagonal are given.
- For finding the area of any polygon its sides and the length of all the principal diagonals are given.

****Read in Detail:**** [****Applications of Heron’s Formula****](https://www.geeksforgeeks.org/maths/applications-of-herons-formula/)

## Heron's Formula Examples

****Example 1: Calculate the area of a triangle whose lengths of sides a, b, and c are 14cm,13cm, and 15 cm respectively.****  
****Solution:****

> ****Given:****    
> a = 14cm  
> b = 13cm  
> c = 15cm
> 
> Firstly, we will determine semi-perimeter(s) ****s = (a + b + c)/2****  
> ⇒ s = (14 + 13 + 15)/2  
> ⇒ s = 21 cm
> 
> Thus, A = √(s(s – a)(s – a)(s – a)  
> ⇒ A = √(21(21 – 14)(21 – 13)(21 – 15)  
> ⇒ A = 84 cm2

****Example 2: Find the area of the triangle if the length of two sides is 11cm and 13cm and the perimeter is 32cm.****  
****Solution:****

> Let a, b and c be the three sides of the triangle.  
> a = 11cm  
> b= 13 cm  
> c = ?
> 
> Perimeter = 32cm
> 
> As we know, Perimeter equals to the sum of the length of three sides of a triangle.  
> ****Perimeter = (a + b + c)****  
> ⇒ 32 = 11 + 13 + c  
> ⇒ c = 32 - 24  
> ⇒ c = 8 cm
> 
> Now as we already know the value of perimeter,  
> ****s = perimeter / 2****  
> ⇒ s = 32 / 2  
> ⇒ s =16 cm
> 
> As, a = 11cm, b = 13 cm, c = 8 cm, s = 16 cm  
> Thus,  A = √(s(s – a)(s – a)(s – a)  
> ⇒ A = √(16(16 – 11)(16 – 13)(16 – 8)  
> ⇒ A = 43.8 cm2

****Example 3: Find the area of an equilateral triangle with a side of 8 cm.****  
****Solution:**** 

> Given,  
> Side = 8 cm
> 
> Area of Equilateral Triangle = √3 / 4 × a2  
> ⇒ Area of Equilateral Triangle = √3 / 4 × (8)2  
> ⇒ Area of Equilateral Triangle = 16 √3 cm2

****Practice Questions****: [Heron's Formula Questions with solutions.](https://www.geeksforgeeks.org/maths/practice-problems-on-herons-formula/)

****Related Reads****

- [Perimeter of the triangle](https://www.geeksforgeeks.org/maths/perimeter-of-a-triangle/)
- [Three Dimensional Geometry](https://www.geeksforgeeks.org/maths/perimeter-of-a-triangle/)
- [Hypotenuse Side Theorem](https://www.geeksforgeeks.org/maths/pythagorean-theorem-formula/#:~:text=Pythagorean%20theorem%20also%20known%20as,the%20longest%20side%20\(hypotenuse\).)
- [Acute Angle Triangle](https://www.geeksforgeeks.org/maths/acute-angled-triangle/#:~:text=An%20acute-angled%20triangle%20is,sum%20property%20of%20the%20triangle.)
- [Right Angled Triangle](https://www.geeksforgeeks.org/maths/right-angled-triangle/#:~:text=Right%20Angled%20Triangle%20Definition,angles%20is%20also%2C%2090%C2%B0.)

We know that the area of a triangle can be calculated using [Heron's Formula](https://www.geeksforgeeks.org/maths/herons-formula/) if all its three side lengths are given.  In the figure given above, ∆ABC is an equilateral triangle with equal sides that measures "a" unit. 

So, AB = BC = CA = a

We know that,

> ****Area of Triangle = √{s(s-a)(s-b)(s-c)}****

where,

- ****s**** is Semi-Perimeter
- ****a****, ****b****, and ****c**** are Side Lengths of Triangle

Here, a = b = c = a

So, s = (a + a + a)/2 = 3a/2

Now, substitute the values in the formula.

A = √{3a/2(3a/2-a)(3a/2-a)(3a/2-a)}

A = √{(3a4)/(4)2}

A = (√3/4) a2

Hence,

> ****Area of Equilateral Triangle = √3/4 a********2****

## Centroid of Equilateral Triangle

Centroid of the triangle also called the centre of the triangle is a point which is at the centre of the triangle. This point is equidistant from all three vertices of the triangle. For an equilateral triangle, as all the sides are equal in length, it is easy to find the centroid for it.

If we draw perpendicular from all the vertices of the equilateral triangle to their opposite sides the point where they all meet is the centroid of the equilateral triangle.

We know that the meeting point of all three perpendiculars of the triangle is called the orthocentre of the triangle. Thus, for an equilateral triangle, the Centroid and Orthocentre are the same points.

For any equilateral triangle ABC its centroid is denoted using point A in the image added below,

![Centroid-of-Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20240124151552/Centroid-of-Triangle.jpg)

> In equilateral triangle with length “a” the distance from the centroid to the vertex is equal to ****√(3a/3)****

## ****Circumcenter of Equilateral Triangle****

The centre of the circle passing through all three vertices of the triangle is called the circumcenter of the triangle. It is calculated by taking the intersection of any two perpendicular bisectors of the triangle.

If the length of the side of the equilateral triangle is aaa, then the ****circumcenter**** is at a distance of:

> Circumradius (R)=Circumradius (R)=a3Circumradius (R)=3​a​

Where:

- a = length of the side of the equilateral triangle.
- R = circumradius, the radius of the circumcircle.

****Note:**** In an Equilateral triangle, the incenter, orthocenter and centroid all coincide with the circumcenter of the equilateral triangle.

![Examples of Equilateral Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230602153030/Examples-of-Equilateral-Triangle-.png)

****Also Check:****

- [Scalene Triangle](https://www.geeksforgeeks.org/maths/scalene-triangle/)
- [Isosceles Triangle](https://www.geeksforgeeks.org/maths/isosceles-triangle/)

## Properties of Equilateral Triangles

Some important characteristics of an equilateral triangle are,

- All three side lengths of an equilateral triangle always measure the same.
- The three interior angles of an equilateral triangle are congruent and equal to 60°.
- According to the angle sum property, the sum of the interior angles of an equilateral triangle is always equal to 180°.
- Equilateral triangles are considered regular polygons since their three side lengths are equal.
- The perpendicular drawn from any vertex of an equilateral triangle bisects the opposite side into two halves. The perpendicular also bisects the angle at the vertex from which it is drawn into 30° each.
- In an equilateral triangle, the orthocenter and centroid are at the same point.
- Median, Angle Bisector and Altitude for all sides of an equilateral triangle are the same.
- [Area of an Equilateral Triangle](https://www.geeksforgeeks.org/maths/area-of-equilateral-triangle/) is √3/4 a2, where "a" is the side length of the triangle.
- [Perimeter of an Equilateral Triangle](https://www.geeksforgeeks.org/maths/how-to-find-the-perimeter-of-an-equilateral-triangle/) is 3a, where "a" is the side length of the triangle.

## Equilateral Triangle Formulas

|Formula for Equilateral Triangles|   |
|---|---|
|Each Interior Angle of Equilateral Triangle|60°|
|Each Exterior Angle of Equilateral Triangle|120°|
|Perimeter of Equilateral Triangle|3 × Sides|
|Height of Equilateral Triangle|√3/2 × (Side)|
|Area of Equilateral Triangle|√3/4 × (Side)2|

## Equilateral Triangle Theorem

Equilateral triangle theorem states that,

> ****"For any equilateral triangle ABC, if P is any point on the arc BC of the circumcircle of the triangle ABC, then PA = PB + PC****
> 
> ****Proof:**** 
> 
> In cyclic quadrilateral ABPC, we have,  
> PA⋅BC = PB⋅AC + PC⋅AB
> 
> As ABC is an equilateral triangle,  
> AB = BC = AC
> 
> Thus,  
> PA.AB = PB.AB + PC.AB
> 
> Simplifying,  
> PA.AB = AB(PB + PC)  
> PA = PB + PC
> 
> Hence, proved.

## Difference Between Scalene, Isosceles, and Equilateral Triangles

Major differences between Scalene Triangle, Isosceles Triangle and Equilateral Triangle is added in the table below,

|Scalene vs Isosceles vs Equilateral Triangles|   |   |
|---|---|---|
|Scalene Triangle|Isosceles Triangle|Equilateral Triangle|
|---|---|---|
|All three side lengths of a scalene triangle are always unequal.|There will be at least two equal side lengths in an isosceles triangle.|All three side lengths of an equilateral triangle always measure the same.|
|All three interior angles of a scalene triangle are always unequal.|The interior angles opposite the equal sides of an isosceles triangle are equal.|The three interior angles of an equilateral triangle are congruent and equal to 60°.|
|![Scalene Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230303184243/Equilateral-Triangle-1.png)|![Isosceles Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230303184346/Isosceles-Triangle-3.png)|![Equilateral Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20230303184252/Equilateral-Triangle-3.png)|

****Also Read:****

> - [How to find the area and perimeter of an equilateral triangle?](https://www.geeksforgeeks.org/maths/how-to-find-the-area-and-perimeter-of-an-equilateral-triangle/)
> - [Triangles in Geometry](https://www.geeksforgeeks.org/maths/triangles/)
> - [Types of Triangles - Equilateral, Isosceles & Scalene](https://www.geeksforgeeks.org/maths/types-of-triangle/)
> - [Examples of Equilateral Triangle in Daily Life](https://www.geeksforgeeks.org/maths/examples-of-equilateral-triangle-in-daily-life/)

## Solved Examples on Equilateral Triangle

****Example 1: Determine the area of an equilateral triangle whose side length is 10 units.****  
****Solution:****

> Given,
> 
> - Side length (a) = 10 units
> 
> We know that,
> 
> Area of Equilateral Triangle = √3/4 a2  
> A = √3/4 × (10)2
> 
> ⇒ A = √3/4 × 100   
> ⇒ A = 25√3 square units ≈ 43.301 square units
> 
> Hence, the area of the given equilateral triangle is approximately equal to 43.301 square units.

****Example 2: Determine the height of an equilateral triangle whose side length is 8 cm.****  
****Solution:****

> Given,
> 
> - Side length (a) = 8 cm
> 
> We know that,
> 
> Height of Equilateral Triangle = √3a/2
> 
> ⇒ H = √3/2 × 8  
> ⇒ H = 4√3 cm  
> ⇒ H ≈ 6.928 cm
> 
> Hence, the height of given equilateral triangle is approximately equal to 6.928 cm.

****Example 3: Determine the perimeter of an equilateral triangle whose side length is 13 cm.****  
****Solution:****

> Given,
> 
> - Side length (a) = 13 cm
> 
> We know that,
> 
> Perimeter of Equilateral Triangle (P) = 3a units  
> ⇒ P = 3 × 13 = 39 cm.
> 
> Hence, the perimeter of the given equilateral triangle is 39 cm.

****Example 4: What is the area of an equilateral triangle if its perimeter is 36 cm?****  
****Solution:****

> Given,
> 
> Perimeter of Equilateral Triangle (P) = 36 cm
> 
> We know that,
> 
> Perimeter of Equilateral triangle (P) = 3a units
> 
> ⇒ 3a = 36  
> ⇒ a = 36/3 = 12 cm
> 
> We know that,
> 
> Area of Equilateral Triangle = √3/4 a2
> 
> ⇒ A = √3/4 × (12)2  
> ⇒ A = √3/4 × 144  
> ⇒ A = 36√3 sq. cm
> 
> Hence, Area of the given equilateral triangle is 36√3 sq. cm.
> 
****Equilateral Triangles****: Equilateral triangles are triangles where all sides and angles are equal. Because all angles are the same, each angle in an equilateral triangle is 60°. Another name for an equilateral triangle is an [equiangular triangle](https://www.geeksforgeeks.org/maths/equilateral-triangle/). Here length of the sides and angles is equal to each other.

****Isosceles Triangles****: An [isosceles triangle](https://www.geeksforgeeks.org/maths/isosceles-triangle/) is a triangle where two sides are equal, and the third side is not equal to the other two. The angles opposite to the equal sides of this triangle are also equal.

****Scalene Triangles****: A [scalene triangle](https://www.geeksforgeeks.org/maths/scalene-triangle/) is one where none of the sides are equal, and none of the angles are equal either. However, the general [properties of triangles](https://www.geeksforgeeks.org/maths/properties-of-triangle/) still apply to scalene triangles. Hence, the sum of all the interior angles is always equal to 180°

### ****Types of Triangles based on angles****

Based on the interior angles of a triangle, we can classify the triangle into three types:

- ****Acute Angled Triangle****
- ****Right Angled Triangle****
- ****Obtuse Angles Triangle****

![Types-of-Triangles-Based-on-Angles](https://media.geeksforgeeks.org/wp-content/uploads/20241205142714882502/Types-of-Triangles-Based-on-Angles-660.webp)

****Acute Angled Triangle****: An [Acute angled Triangle](https://www.geeksforgeeks.org/maths/acute-angled-triangle/) is one where all the interior angles of the Triangle are less than 90°. For Instance, an Equilateral Triangle is an acute-angled triangle (all angles are less than 90°).

****Right Angled Triangle****: A [Right Angled Triangle](https://www.geeksforgeeks.org/maths/right-angled-triangle/) is one where one of the angles is always equal to 90°. Pythagoras' Theorem applies to right-angled triangles. It says that the square of the hypotenuse (the longest side) equals the sum of the squares of the base and perpendicular.

 ****Obtuse Angled Triangle****: In an [obtuse-angled triangle](https://www.geeksforgeeks.org/maths/obtuse-angled-triangle/), one angle measures more than 90°. Here, one of the three angles is greater than 90°, making the other two angles less than 90°.

## Classification Based on the Sides and Angles of Triangle

There are various other types of triangles based on both angles and sides of the triangle, some of these types are:

- Isosceles Right Triangle
- Obtuse Isosceles Triangle
- Acute Isosceles Triangle
- Right Scalene Triangle
- Obtuse Scalene Triangle
- Acute Scalene Triangle

### For Isosceles Triangle:

  

![type2](https://media.geeksforgeeks.org/wp-content/uploads/20241216181034713762/type2.webp)

****Isosceles Right Triangle****: An isosceles right triangle, also called a right isosceles triangle, has two main characteristics. First, it possesses two sides of equal length, often referred to as the legs. Second, it contains one angle measuring exactly 90 degrees, known as the right angle.

> ****For example****, look at triangle ABC. Sides AB and BC both measure 8 centimeters (AB = BC = 8cm), and angle B is a right angle (∠B = 90°). It's an Isosceles Right Triangle.

****Obtuse Isosceles Triangle****: An Obtuse Isosceles Triangle is a triangle with two sides equal and one interior angle measuring more than 90°.

> ****For Example****, in triangle PQR, sides PQ = PR = 10 cm, and ∠P = 110°, it's an Obtuse Isosceles Triangle.

****Acute Isosceles Triangle****: An Acute Isosceles Triangle is a triangle with two equal sides, and all interior angles measuring less than 90°.

> ****For Example****, in triangle LMN, sides LM = LN = 8 cm, and ∠M = ∠N = 70°, it's an Acute Isosceles Triangle.

### For Scalene Triangle:

![type1](https://media.geeksforgeeks.org/wp-content/uploads/20241216181447340561/type1.webp)

****Right Scalene Triangle****: A Right triangle is a triangle with all sides of different lengths and one interior angle to be 90°.

> ****For example****, in triangle ABC, sides AC = 6 cm, BC = 8 cm and AB = 10 cm. Plus, angle C measures 90 degrees, making it a perfect example of a Right Scalene Triangle.

****Obtuse Scalene Triangle:**** An Obtuse Scalene Triangle is a triangle with all sides of different lengths and one obtuse angle.

> ****For example,**** in triangle DEF where all three sides are different lengths. On top of that, one of the angles inside, ∠F, opens extra wide at 135 degrees, more than a right angle! That's what makes DEF an Obtuse Scalene Triangle.

****Acute Scalene Triangle****: An Acute Scalene Triangle is a triangle with all angles less than a right angle and all sides of different lengths.

> ****For Example****: in triangle GHI, where each side is of different length: GH = 5 cm, HI = 7 cm and IG = 9 cm. And all three angles are sharp, none wider than a right angle with ∠I = 60 degrees. That makes GHI an Acute Scalene Triangle.

****Also, Read****

> - [Similar Triangles](https://www.geeksforgeeks.org/maths/similar-triangles/)
> - [Area of Triangle](https://www.geeksforgeeks.org/maths/similar-triangles/)
> - [Perimeter of a Triangle](https://www.geeksforgeeks.org/maths/how-to-find-the-perimeter-and-area-of-a-triangle/)
> - [Polygon](https://www.geeksforgeeks.org/maths/polygon-formula/)

Based on sides, there are 3 [types of triangles](https://www.geeksforgeeks.org/maths/types-of-triangle/):

1. Scalene Triangle
2. Isosceles Triangle
3. Equilateral Triangle

![types-of-triangle-based-on-sides-min](https://media.geeksforgeeks.org/wp-content/uploads/20241205142924279323/types-of-triangle-based-on-sides-min.jpg)

Types of Triangles based on Side

  

****Equilateral Triangle****

In an [Equilateral triangle](https://www.geeksforgeeks.org/maths/equilateral-triangle/), all three sides are equal to each other as well as all three interior angles of the equilateral triangle are equal.  
Since all the interior angles are equal and the sum of all the interior angles of a triangle is 180° (one of the Properties of the Triangle). We can calculate the individual angles of an equilateral triangle.

∠A+ ∠B+ ∠C = 180°  
∠A = ∠B = ∠C

Therefore, 3∠A = 180°

∠A= 180/3 = 60°  
Hence, ∠A = ∠B = ∠C = 60°

****Properties of Equilateral Triangle****

- All sides are equal.
- All angles are equal and are equal to 60°
- There exist three lines of symmetry in an equilateral triangle
- The angular bisector, altitude, median, and perpendicular line are all the same and here it is AE.
- The orthocentre and centroid are the same.

****Equilateral Triangle Formulas****

The basic formulas for equilateral triangles are:

> - [Area of Equilateral Triangle](https://www.geeksforgeeks.org/maths/area-of-equilateral-triangle/) = ****√3/4 × a********2****
> - Perimeter of Equilateral Triangle = ****3a****
> 
> ****where****, ****a**** is Side of Triangle

****Isosceles Triangle****

In an [Isosceles triangle](https://www.geeksforgeeks.org/maths/isosceles-triangle/), two sides are equal and the two angles opposite to the sides are also equal. It can be said that any two sides are always congruent. The area of the [Isosceles triangle](https://www.geeksforgeeks.org/maths/area-of-isosceles-triangle/) is calculated by using the formula for the area of the triangle as discussed above.

****Properties of Isosceles Triangle****

- Two sides of the isosceles triangle are always equal
- The third side is referred to as the base of the triangle and the height is calculated from the base to the opposite vertex
- Opposite angles corresponding to the two equal sides are also equal to each other.

### ****Scalene Triangle****

In a [Scalene triangle](https://www.geeksforgeeks.org/maths/scalene-triangle/), all sides and all angles are unequal. Imagine drawing a triangle randomly and none of its sides is equal, all angles differ from each other too.

****Properties of Scalene Triangle****

- None of the sides are equal to each other.
- The interior angles of the scalene triangle are all different.
- No line of symmetry exists.
- No point of symmetry can be seen.
- Interior angles may be acute, obtuse, or right angles in nature (this is the classification based on angles).
- The smallest side is opposite the smallest angle and the largest side is opposite the largest angle (general property).

### ****Types of Triangles Based on Angles****

Based on angles, there are 3 types of triangles:

1. Acute Angled Triangle
2. Obtuse Angled Triangle
3. Right Angled Triangle

![Types-of-Triangles-Based-on-Angles](https://media.geeksforgeeks.org/wp-content/uploads/20241205142714882502/Types-of-Triangles-Based-on-Angles.webp)

Types of Triangles based on Angles

  

****Acute Angled Triangle****

In [Acute angle triangles](https://www.geeksforgeeks.org/maths/acute-angled-triangle/), all the angles are greater than 0° and less than 90°. So, it can be said that all 3 angles are acute (angles are less than 90°)

****Properties of Acute Angled Triangles****

- All the interior angles are always less than 90° with different lengths of their sides.
- The line that goes from the base to the opposite vertex is always perpendicular.

****Obtuse Angled Triangle****

In an [obtuse angle Triangle](https://www.geeksforgeeks.org/maths/obtuse-angled-triangle/), one of the 3 sides will always be greater than 90°, and since the sum of all three sides is 180°, the rest of the two sides will be less than 90° (angle sum property).

****Properties of Obtuse Angled Triangle****

- One of the three angles is always greater than 90°.
- The sum of the remaining two angles is always less than 90° (angle sum property).
- Circumference and the orthocentre of the obtuse angle lie outside the triangle.
- The incentre and centroid lie inside the triangle.

****Right Angled Triangle****

When one angle of a triangle is exactly 90°, then the triangle is known as the [Right Angle Triangle](https://www.geeksforgeeks.org/maths/right-angled-triangle/). 

****Properties of Right-angled Triangle****

- A Right-angled Triangle must have one angle exactly equal to 90°, it may be scalene or isosceles but since one angle has to be 90°, hence, it can never be an equilateral triangle.
- The side opposite 90° is called Hypotenuse.
- Sides are adjacent to the 90° are base and perpendicular.
- ****Pythagoras Theorem:**** It is a special property for Right-angled triangles. It states that the square of the hypotenuse is equal to the sum of the squares of the base and perpendicular i.e. ****AC********2**** ****= AB********2**** ****+ BC********2****

## Triangle Formulas

In geometry, for every two-dimensional shape (2D shape), there are always two basic measurements that we need to find out, i.e., the area and perimeter of that shape. Therefore, the triangle has two basic formulas which help us to determine its area and perimeter. Let us discuss the formulas in detail.

### Perimeter of Triangle

The [Perimeter of a triangle](https://www.geeksforgeeks.org/maths/perimeter-of-a-triangle/) is the total length of all its sides. It can be calculated by adding the lengths of the three sides of the triangle, suppose a triangle with sides a, b, and c is given then its perimeter is given by****:****

> Perimeter of triangle = a + b + c

![Perimeter of Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20240319170630/triangle-perimeter.webp)

Perimeter of a Triangle

### Area of a Triangle

The [area of a triangle](https://www.geeksforgeeks.org/maths/area-of-triangle/) is the total area covered by the triangle boundary. It is equal to half the product of its base and height. It is measured in square units. If the base of a triangle is ****b**** and its height is ****h**** then its area is given by:

> Area of Triangle = 1/2 × b × h

![Area of Triangle](https://media.geeksforgeeks.org/wp-content/uploads/20240319170736/triangle-area.webp)

Area of a Triangle

****Read More:**** [****Finding the Area of the Triangle using Heron's Formula****](https://www.geeksforgeeks.org/maths/herons-formula/)****.****  
****Also check****: [****How to Find Area of Triangle, Formulas, Examples****](https://www.geeksforgeeks.org/maths/area-of-triangle/)****.****

## ****Properties of Triangles****

Various properties of triangles are,

- A Triangle has 3 sides, 3 vertices, and 3 angles.

- [****Area of Triangle****](https://www.geeksforgeeks.org/maths/area-of-triangle/)****:**** 1/2× base × height.

- For similar triangles, the angles of the two triangles have to be congruent to each other and the respective sides should be proportional.

- The difference between the length of any two sides is always lesser than the third side. For example, AB - BC < AC or BC - AC < AB

- The side opposite the largest angle is the largest side of the triangle. For instance, in a right-angled triangle, the side opposite 90° is the longest.

- The sum of the length of any two sides of a triangle is always greater than the third side. For example, AB + BC > AC or BC + AC > AB.

- ****Angle Sum Property:**** The sum of all three interior angles is always 180°. Therefore. In the Triangle ΔPQR shown above, ∠P + ∠Q + ∠R = 180°, the interior angles of a triangle will be greater than 0° and less than 180°. 

- The perimeter of a figure is defined by the overall length the figure is covering. Hence, the perimeter of a triangle is equal to the sum of lengths on all three sides of the triangle. Perimeter of ΔABC= (AB + BC + AC)

- The exterior angle of a triangle is equal to the sum of the interior opposite and non-adjacent angles (also referred to as remote interior angles). In the triangle ΔPQR, if we extend side QR to form an exterior angle ∠PRS, then: ∠PRS = ∠PQR + ∠PRQ

****Read More about the**** [****Properties of Triangles****](https://www.geeksforgeeks.org/maths/properties-of-triangle/)****.****

## Important Concepts of Triangle

Triangle is an important chapter, let's learn the important concepts of Triangle.

- ****Median of Triangle****  
    A median is a line segment that joins the vertex of the triangle with the midpoint of opposite sides of the triangle. A median of a triangle bisects the side which it joins.

- ****Altitude of Triangle****  
    The altitude of the Triangle is the perpendicular distance from the base of the triangle to its apex vertex.

- ****Centroid of Triangle****  
    A centroid is the point inside a triangle where all the medians of a triangle meet each other. The Centroid of a Triangle divides the median into 2:1.

- ****Circumcentre of a Triangle****  
    The Circumcentre of a Triangle is the point where all the perpendicular bisectors of the sides of the triangle.

- ****Orthocentre of a Triangle****  
    The Orthocentre of a Triangle is the point where all the altitudes of a triangle meet each other.

- ****Incentre of a Triangle****  
    The center of a Triangle is a point where all the angle bisectors of a triangle meet each other.

## Fun Facts about Triangles

Below are 10 interesting facts about Triangles that find real-life significance:

- ****Strong and Stable****: Triangles are incredibly strong for their size because they distribute stress evenly throughout their shape. This is why bridges, trusses, and even airplane wings are often built using triangular frameworks.

- ****Minimal Materials****: Because triangles are so strong, they can be built using less material compared to other shapes for the same level of stability. This makes them a lightweight and efficient choice in construction.

- ****Stacking Efficiency****: Triangles, especially equilateral ones, can be packed together very efficiently. This is useful for things like creating stable and space-saving containers for fruits and vegetables.

- ****Direction Indicators****: Triangles are universally recognized as pointing arrows. This makes them ideal for road signs, warning labels, and directional markers, ensuring clear communication.

- ****Musical Harmony****: The basic principles of harmony in music rely on the perfect fifth, which has a frequency ratio of 3:2. This ratio can be visualized as a 30-60-90 degree triangle, making triangles a foundational concept in musical theory.

- ****Tooth Shape Efficiency****: Our premolar teeth have triangular cusps that are perfect for grinding and tearing food. The triangular shape allows for maximum surface area and efficient chewing.

- ****Aerodynamic Design****: The triangular shape plays a role in aerodynamics. The delta wing, a triangular airplane wing design, is known for its stability and maneuverability at high speeds.

- ****Facial Recognition****: Our brains use triangles to recognize faces. The triangular arrangement of eyes, nose, and mouth helps us quickly identify and differentiate faces.

- ****Fracture Lines****: Even in breaking, triangles can be helpful! Cracks in glass or other materials often propagate in triangular patterns, which can help predict how something might break and potentially prevent accidents.

- ****Building Blocks of Life****: The basic building block of DNA, the double helix, can be visualized as two intertwined triangles representing the sugar-phosphate backbones. This triangular structure is essential for storing and transmitting genetic information.

## Triangles Solved Examples

****Example 1: In a triangle ∠ACD = 120°, and ∠ABC = 60°. Find the type of Triangle.****  
****Solution:****

> In the above figure, we can say, ∠ACD = ∠ABC + ∠BAC (Exterior angle Property)
> 
> 120° = 60° + ∠BAC  
> ∠BAC = 60°  
> ∠A + ∠B + ∠C = 180°  
> ∠C OR ∠ACB = 60°
> 
> Since all the three angles are 60°, the ****triangle is an Equilateral Triangle.****

****Example 2: The triangles with sides of 5 cm, 5 cm, and 6 cm are given. Find the area and perimeter of the Triangle.****  
****Solution:**** 

> Given, the sides of a triangle are 5 cm, 5 cm, and 6 cm  
> Perimeter of the triangle = (5 + 5 + 6) = 16 cm  
> Semi Perimeter = 16 / 2 = 8 cm  
> Area of Triangle = √s(s - a)(s - b)(s - c) (Using Heron's Formula)  
> = √8(8 - 5)(8 - 5)(8 - 6)  
> = √144 = 12 cm2

****Example 3: In the Right-angled triangle, ∠ACB = 60°, and the length of the base is given as 4cm. Find the area of the Triangle.****  
****Solution:****

> Using trigonometric formula of tan60°,  
> tan60° = AB / BC = AB  /4  
> AB = 4√3cm  
> Area of Triangle ABC = 1/2   
> = 1/2 × 4 × 4√3   
> = 8√3 cm2

****Example 4: In ΔABC if ∠A + ∠B = 55°. ∠B + ∠C = 150°, Find angle B separately.****  
****Solution:****

> Angle Sum Property of a Triangle says ∠A + ∠B + ∠C= 180°  
> Given:   
> ∠A + ∠B = 55°  
> ∠B + ∠C = 150°
> 
> Adding the above 2 equations,
> 
> ∠A + ∠B + ∠B + ∠C= 205°  
> 180° + ∠B= 205°  
> ∠B = 25°

****Articles related to Triangles:****

> - [Congurency of triangles](https://www.geeksforgeeks.org/maths/congruence-of-triangles/)
> - [Similar Triangles](https://www.geeksforgeeks.org/maths/similar-triangles/)
> - [Angle Sum Property](https://www.geeksforgeeks.org/maths/angle-sum-property-triangle/) 

![Triangles in a Star](https://media.geeksforgeeks.org/wp-content/uploads/20230416200047/Triangle-2.png)

> A star has 10 triangles that are,
> 
> ∆ABH, ∆BIC, ∆CJD, ∆DFE, ∆EGA, ∆DGI, ∆GJB, ∆FAI, ∆JGB and ∆EHJ



a right-angled [triangle](https://www.geeksforgeeks.org/maths/triangles/), the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides.
> 
> ****hypotenuse********2**** ****= perpendicular********2**** ****+ base********2****

It provides us with the relationship between the sides in a right-angled triangle. A right triangle consists of two legs and a hypotenuse.

## Pythagoras Theorem Formula

Pythagoras theorem formula is AC2 = AB2 + BC2, where AB is the perpendicular side, BC is the base, and AC is the hypotenuse side. The Pythagoras equation is applied to any right-angled triangle, a triangle that has one of its angles equal to 90°.

![Pythagoras Theorem Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20220802114107/PythagorasTheorem.png)

Pythagoras Formula

The three sides of the right-angled triangle are called the [Pythagoras Triplets.](https://www.geeksforgeeks.org/maths/pythagorean-triples/) 

## Pythagoras Theorem Proof

Consider a right-angled triangle having sides A, B, and C. Here, AC is the longest side (hypotenuse), and AB and BC are the legs of the triangle. Draw a perpendicular line BD at AC as shown in the figure below,

![Pythagorus Theorem Proof Illustration](https://media.geeksforgeeks.org/wp-content/uploads/20220802113804/DerivationPythagoras-660x384.png)

Derivation of Pythagoras Theorem

> In △ABD and △ACB,
> 
> ∠A = ∠A (Common angle)
> 
> ∠ADB = ∠ABC (90°)
> 
> Therefore, we can say △ABD ∼ △ ACB (By AA Similarity)
> 
> Similarly, △BDC ∼ △ACB
> 
> Hence, AD/AB = AB/AC
> 
> AB2 = AD × AC ⇢ (1)
> 
> And, CD/BC = BC/AC
> 
> BC2 = CD × AC ⇢ (2)
> 
> Adding equations (1) and (2),
> 
> AB2 + BC2 = AC × AD + AC × CD
> 
> AB2 + BC2 = AC (AD + CD)
> 
> AB2 + BC2 = AC × AC
> 
> AB2 + BC2 = AC2
> 
> Also, AC2 = AB2 + BC2
> 
> Hence proved.

## Converse of Pythagoras Theorem

****The converse of the Pythagoras theorem states that,****

> Given a triangle with sides of length a, b, and c, if a2 + b2 = c2, then the angle between sides a and b is a right angle. For any three positive real numbers a, b, and c such that a2 + b2 = c2, there exists a triangle with sides a, b and c as a consequence of the converse of the triangle inequality.

### ****Converse of Pythagoras Theorem Proof****

For a triangle with the length of its sides a, b, if ****c********2**** ****= a********2**** ****+ b********2****, we need to prove that the triangle is right-angled.

![Converse of Pythagoras Theorem](https://media.geeksforgeeks.org/wp-content/uploads/20220802121232/Converse1-660x390.png)

We assume that it satisfies ****c********2**** ****= a********2**** ****+ b********2********,**** and by looking into the diagram, we can tell that ∠C = 90°, but to prove it, we require another triangle △EGF, such as AC = EG = b and BC = FG = a.

![Converse of Pythagorean Theorem Proof](https://media.geeksforgeeks.org/wp-content/uploads/20220802121314/Converse2-660x390.png)

> In △EGF, by Pythagoras Theorem:
> 
> ⇒ EF2 = EG2 + FG22 = b2 + a2 ⇢ (1)
> 
> In △ABC, by Pythagoras Theorem:
> 
> ⇒ AB2 = AC2 + BC2 = b2 + a2 ⇢ (2)
> 
> From equation (1) and (2), we have;
> 
> ⇒ EF2 = AB2
> 
> ⇒ EF = AB
> 
> ⇒ △ ACB ≅ △EGF (By SSS postulate)
> 
> ⇒ ∠G is right angle
> 
> Thus, △EGF is a right triangle. Hence, we can say that the converse of the Pythagorean theorem also holds.

## History of Pythagoras Theorem

The history of the Pythagoras Theorem goes back to the ancient Babylon and Egypt eras. It is named after the ancient Greek mathematician and philosopher Pythagoras of Samos. He lived during the 6th century BCE.

But the roots of this theorem go to ancient cultures. It is very likely that Babylonians and Indians used this theorem well before Pythagoras, but its widespread use came into existence after Pythagoras stated it. The Pythagorean theorem is also known as the Baudhayana theorem, listed in the book Baudhāyana Śulbasûtra by the great Indian mathematician Baudhāyana.

One of the other reasons this theorem is known as Pythagoras or Pythagorean Theorem is because the disciples of Pythagoras spread knowledge and philosophy of Pythagoras after his death as well.

![Proof of Pythagoras Theorem](https://media.geeksforgeeks.org/wp-content/uploads/20220802112920/Pythagoras-660x281.png)

## Pythagoras Theorem Applications

Below are some of the uses of Pythagorean Theorem in real life:

- ****Solving Right-Angled Triangles****: Students use the theorem to calculate the length of unknown sides of right-angled triangles, given the other two sides. This is crucial for solving problems in trigonometry, mensuration, and coordinate geometry.
- ****Distance and Height Problems****: The theorem is applied in real-world problems, such as finding the height of buildings or the distance between two points on a plane, helping students understand practical applications.
- ****Proofs and Derivations****: Students also learn to prove Pythagoras’ Theorem and use it to derive other important mathematical properties, such as the distance formula in coordinate geometry.
- ****Vectors and 3D Geometry****: Pythagoras' Theorem is applied in calculating the magnitude of vectors and distances between points in three-dimensional space.
- ****Calculus and Coordinate Geometry****: The theorem underpins the derivation of formulas used in differential calculus and helps solve problems involving curves, tangents, and distances between geometric figures.
- ****Physics Applications****: In physics, students use Pythagoras' Theorem to solve problems involving forces, velocities, and resultant vectors.
- ****Construction and Architecture****: Builders and architects use Pythagoras' Theorem to ensure accurate measurements when constructing buildings, bridges, or other structures. It helps in creating right angles and calculating diagonal lengths when designing layouts and foundations.
- ****Navigation****: In navigation, the theorem is used to calculate the shortest distance between two points on a map, which is essential for both air and sea travel. By treating distances as right-angled triangles, navigators can find the direct route between locations.

### ****People Also Read:****

> - [Angles in Right Angled Triangle](https://www.geeksforgeeks.org/maths/how-to-find-an-angle-in-a-right-angled-triangle/)
> - [Right Angled Triangle Formula](https://www.geeksforgeeks.org/maths/right-angled-triangle/)
> - [Similar Triangles](https://www.geeksforgeeks.org/maths/similar-triangles/)
> - [Congruence of Triangles](https://www.geeksforgeeks.org/maths/congruence-of-triangles/)

## Solved Examples on Pythagoras Theorem

Let's solve some questions on Pythagoras Theorem.

****Example 1: In the below given right triangle, find the value of y.****

![Pythagoras Theorem Example](https://media.geeksforgeeks.org/wp-content/uploads/20220802121523/Pythagoras1-660x390.png)

****Solution:**** 

> By the statement of the Pythagoras theorem we get,
> 
> ⇒ z2 = x2 + y2
> 
> Now, substituting the values directly we get,
> 
> ⇒ 132 = 52 + y2
> 
> ⇒ 169 = 25 + y2
> 
> ⇒ y2 = 144
> 
> ⇒ y = √144 = 12

****Example 2:**** ****Given a rectangle with a length of 4 cm and breadth of 3 cm. Find the length of the diagonal of the rectangle.****

****Solution:**** 

> ![Solved Example of Pythagoras Theorem](https://media.geeksforgeeks.org/wp-content/uploads/20220802121630/Pythagoras2-660x390.png)
> 
> In the above diagram length of the rectangle is 4 cm, and the width is 3 cm. Now we have to find the distance between point A to point C or point B to point D. Both give us the same answer because opposite sides are of the same length i.e., AC = BD. Now let's find the distance between points A and C by drawing an imaginary line.
> 
> ![Pythagorean Theorem Example](https://media.geeksforgeeks.org/wp-content/uploads/20220802121715/Pythagoras3-660x390.png)
> 
> Now triangle ACD is a right triangle. 
> 
> So by the statement of Pythagoras theorem,
> 
> ⇒ AC2 = AD2 + CD2
> 
> ⇒ AC2 = 42 + 32
> 
> ⇒ AC2 = 25
> 
> ⇒ AC = √25 = 5
> 
> Therefore length of the diagonal of given rectangle is 5 cm.

****Example 3: The sides of a triangle are 5, 12, and 13. Check whether the given triangle is a right triangle or not.****

****Solution:**** 

> Given,
> 
> ⇒ a = 5
> 
> ⇒ b = 12
> 
> ⇒ c = 13
> 
> By using the converse of Pythagorean Theorem,
> 
> ⇒ a2 + b2 = c2
> 
> Substitute the given values in the above equation,
> 
> ⇒ 132 = 52 + 122
> 
> ⇒ 169 = 25 + 144
> 
> ⇒ 169 = 169
> 
> So, the given lengths satisfy the above condition.
> 
> Therefore, the given triangle is a right triangle.

****Example 4:**** ****The side of a triangle is of lengths 9 cm, 11 cm, and 6 cm. Is this triangle a right triangle? If so, which side is the hypotenuse?****

****Solution:**** 

> We know that hypotenuse is the longest side. If 9 cm, 11 cm, and 6 cm are the lengths of the angled triangle, then 11 cm will be the hypotenuse.
> 
> Using the converse of Pythagoras theorem, we get
> 
> ⇒ (11)2 = (9)2 + (6)2
> 
> ⇒ 121 = 81 + 36
> 
> ⇒ 121 ≠ 117
> 
> Since, both the sides are not equal therefore 9 cm, 11 cm and 6 cm are not the side of the right-angled triangle.

## Practice Questions on Pythagoras Theorem

Here are some practice questions on the Pythagoras theorem for you to solve.

****Q1: If the two shorter sides of a right angled triangle measures 14 and 15 cm, find the length of the longest side.****

****Q2: If the hypotenuse and perpendicular of a right angled triangle are 5 and 4 cm then find the base.****

****Q3: In a triangular field the sides measures 24 cm, 7 cm and 25 cm then prove that field is the form of a right triangle.****

****Q4: A wall is of 12 m height and a ladder of 13 m is placed against it touching its top. Find the distance between the foot of the ladder and the wall.****

## ****Conclusion****

Pythagorean Theorem is one of the most important concepts in mathematics. It helps us understand the relationship between the sides of a right-angled triangle and is widely used in various fields such as geometry, construction, physics, and even navigation. By knowing just two sides of a right-angled triangle, we can easily calculate the third, making it a powerful tool for solving real-world problems. This simple formula makes finding accurate solutions much easier in various everyday tasks, such as measuring objects, calculate distances etc.



In ****Figure 1**** the rectangle ABCD has the diagonal ****AC**** and its length can be calculated using the [Pythagoras theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/). The diagonal of the rectangle are equal and they bisect each other.

### ****Diagonal of Rectangle Formula****

If the ****length**** and the ****breadth**** of the rectangle are l and b respectively then the ****diagonal of the rectangle**** is 

> ****d = √( l********2**** ****+ b********2********)****

## Rectangle Properties

Some of the important properties of a rectangle are:

- Rectangle is a Polygon with four sides.
- The angle formed by adjacent sides is 90°.
- Rectangle is a [Quadrilateral](https://www.geeksforgeeks.org/maths/quadrilateral/).
- Opposite sides of a Rectangle are Equal and Parallel.
- Rectangle is considered to be a Parallelogram.
- Interior Angles of a Rectangle are equal and the value of each angle is 90°.
- Sum of all Interior Angles is 360°.
- Diagonals of a Rectangle are equal.
- Diagonals of a Rectangle bisect each other.
- Length of the diagonal is found by Pythagoras theorem. If the length and the breadth of the rectangle are l and b respectively then the diagonal of the rectangle is d = √( l2 + b2).

## Rectangle Formula

A rectangle is a closed quadrilateral in which the sides are equal and parallel, there are various formulas that are used to find various parameters of the rectangle.  
Some of the important formulas of the rectangle are,

1. Perimeter of Rectangle
2. Area of Rectangle

|****Property****|****Formula****|
|---|---|
|Area|Length × Breadth|
|Perimeter|2 × (Length + Breadth)|

Let's discuss them in some detail.

## Rectangle Perimeter

Perimeter of the rectangle is defined as the total length of all the sides of the rectangle. It is also called the circumference of the rectangle.  
It is measured in units of length, i.e. m, cm, etc.

### Perimeter of Rectangle Formula

In a rectangle, if the length(l) and breadth(b) are given then the perimeter of the rectangle is found using the formula,

> [****Perimeter of Rectangle****](https://www.geeksforgeeks.org/maths/perimeter-of-rectangle/) ****= 2 (l + b)****
> 
> ****their****

Where,

- ****l**** is the length of the rectangle.
- ****b**** is the breadth of the rectangle.

The figure representing the perimeter of the rectangle is,

![Rectangle Perimeter Formula](https://media.geeksforgeeks.org/wp-content/uploads/20230713172219/Perimeter-of-Rectangle.png)

Perimeter of Rectangle Illustration

### Length of Rectangle Formula

In a rectangle, if the ****breadth(b)**** and ****perimeter(P)**** are given then its ****length(l)**** is found using,

> ****(length) l = (P / 2) - b****

In a rectangle, if the ****breadth(b)**** and ****area (A)**** are given then its ****length(l)**** is found using,

> ****(length) l = A/b****

### Formula for Breadth of Rectangle

In a rectangle, if the ****length(l)**** and ****perimeter(P)**** are given then its ****breadth(b)**** is found using,

> ****(breadth) b = (P / 2) - l****

In a rectangle, if the ****length(l)**** and ****area (A)**** are given then its ****breadth(b)**** is found using,

> ****(breadth) b = A/l****

## Rectangle Area

Area of the rectangle is defined as the total space occupied by the rectangle. It is the space inside the boundary or the perimeter of the rectangle.

The area of the rectangle is dependent on the length and breadth of the rectangle, it is the product of the length and the breadth of the rectangle. The area of a rectangle is measured in square units, i.e. in m2, cm2, etc.

### Area of Rectangle Formula

In a rectangle, if the ****length(l)**** and ****breadth(b)**** are given then the ****area of the rectangle**** is found using the formula,

> [****Area of Rectangle****](https://www.geeksforgeeks.org/maths/area-of-rectangle/) ****= l × b****
> 
> where,
> 
> - ****l**** is the length of the rectangle
> - ****b**** is the breadth of the rectangle

Here is a diagram representing the area of the rectangle is,

![Calculating Rectangle Area](https://media.geeksforgeeks.org/wp-content/uploads/20230713172031/Area-of-Rectangle.png)

Area of Rectangle Diagram

## Types of Rectangle

Rectangles are of two types, which are:

- Square
- Golden Rectangle

Let's learn about them in detail.

### Square

> A square is a special case of a rectangle in which all the sides are also equal. It is a quadrilateral in which opposite sides are parallel and equal, so it can be considered a rectangle.

The diagram of a square is shown below,

![Square Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20230717091051/Square.png)

Diagram of a Square

****Learn More:**** [****Square****](https://www.geeksforgeeks.org/maths/square/)

### Golden Rectangle

A rectangle in which the 'length to the width' ratio is similar to the golden ratio, i.e. equal to the ratio of 1: (1+⎷5)/2 is called the golden rectangle.   
They are in the ratio of 1: 1.618 thus, if its width is 1 m then its length is 1.168 m.

The diagram of a Golden Rectangle is shown below,

![Golden Rectangle Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20230717091109/Golden-rectabgle.png)

****Must Read****

> - [****Triangle****](https://www.geeksforgeeks.org/maths/triangles/)
> - [****Rhombus****](https://www.geeksforgeeks.org/maths/rhombus/)
> - [****Trapezium****](https://www.geeksforgeeks.org/maths/trapezium/)

## ****Examples on Rectangle****

Here are some solved examples on the basic concepts of rectangle for your help.

****Example 1: Find the area of the rectangular photo frame whose sides are 8 cm and 6 cm.****

****Solution:****

> Given,
> 
> - Length of Photo Frame = 8 cm
> - Breadth of Photo Frame = 6 cm
> 
> Area of Photo Frame = Length × Breadth  
> = 6 × 8 = 48 cm2
> 
> Thus, the area of the photo frame is 48 cm2.

****Example 2: Find the perimeter of the rectangular field whose sides are 9 m and 13 m.****

****Solution:****

> Given,
> 
> - Length of Rectangular Field = 9 m
> - Breadth of Rectangular Field = 13 m
> 
> Perimeter of Rectangular Field = 2 × (l + b)  
> = 2 × (9 + 13) = 2 × (22)  
> = 44 m
> 
> Thus, the perimeter of the rectangular field is 44 m.

****Example 3: Find the area and the perimeter of the room with a length of 12 feet and a breadth of 8 feet.****

****Solution:****

> Given,
> 
> - Length of room (l) = 12 feet
> - Breadth of room (b) = 8 feet
> 
> Area of Room = Length × Breadth  
> = 12 × 8 = 96 feet2
> 
> Perimeter of Room = 2 × (l + b)  
> = 2 × (12 + 8) = 2 × 20  
> = 40 feet
> 
> Thus, the area of the room is 96 feet2, and the perimeter is 40 feet.

****Example 4: Find the length of the diagonal of the rectangle whose sides are 6 cm and 8 cm.****

****Solution:****

> Given,
> 
> - Length of rectangle (l) = 6 cm
> - Breadth of rectangle (b) = 8 cm
> 
> Length of diagonal (d) = √( l2 + b2)
> 
> d = √( 82 + 62)   
> d = √ (64 + 36) = √(100)  
> d = 10 cm
> 
> Thus, the length of the rectangle is 10 cm


- It is a type of [rectangle](https://www.geeksforgeeks.org/maths/rectangle/) where the length and width are the same.
- A square also has the property that its diagonals are equal in length and bisect each other at right angles.
- The ****perimeter**** of a square is 4 x a where a is length of a side.
- Square of area is a2

![Perimeter of Square Formula](https://media.geeksforgeeks.org/wp-content/uploads/20230310130236/Perimeter-of-Square-2.png)

### Visual Representation of a Perimeter of a Square

![Square Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20230310130701/Perimeter-of-Square.png)

- ****Edges****: The four equal sides of the square are highlighted, representing the boundaries.
- ****Diagonal****: The diagonal of the square is shown, connecting opposite vertices.
- ****Angles****: All four angles will be right angles (90°).
- ****Vertex****: The corners of the square, where two edges meet, are labeled as vertices.
- ****Perimeter (Blue Color)****: The perimeter is illustrated in blue, emphasizing the total length of the boundary.
- ****Area (Green Color)****: The area of the square is shown in green, representing the space enclosed within the square.

## Perimeter of Square Formula

The perimeter of the Square can be calculated using the following formula:

> ****Perimeter of square = 4 × side****

The unit for the perimeter of a square is the unit used for length. It is measured in meters (m), centimeters (cm), inches (in), etc.

## Various Methods to Find Perimeter of Square

Perimeter of square can be calculated using these three methods:

- Using Side length
- Using Diagonal
- Using Area

### The Perimeter of the ****the**** Square using Side Length

The below-given steps can be used to find the perimeter of square using side length,

> ****Step 1:**** Measure the side of the square.  
> ****Step 2:**** Multiply the side length by 4.  
> ****Step 3:**** Express the perimeter obtained in the respective unit.

### Perimeter of Square using Diagonal

If the side of the square is not given, but t****he diagonal**** is given, but the diagonal is given, then we make use of a different formula.

The of a square is given by the following formula :

![Diagonal of Square Formula](https://media.geeksforgeeks.org/wp-content/uploads/20230310130342/Perimeter-of-Square-3.png)

Diagonal of Square

The formula for the Perimeter of square using the diagonal is given below.

> ****P = 4 × (Diagonal/√2).****

****Read More:**** [Diagonal of the square](https://www.geeksforgeeks.org/maths/diagonal-of-a-square-formula/).

### Perimeter of Square using Area

When the area of square is given, let us assume the [Area of the ****the**** square](https://www.geeksforgeeks.org/maths/area-of-a-square/) is ****a****. As we all know, Area, a = (side)2  
Therefore, the perimeter of the square is,

![Perimeter of Square Formula using Area](https://media.geeksforgeeks.org/wp-content/uploads/20230310130502/Perimeter-of-Square-5.png)

****Related Reads****:

> - [Perimeter of Rectangle](https://www.geeksforgeeks.org/maths/perimeter-of-rectangle/)
> - [Perimeter of Geometrical Shapes](https://www.geeksforgeeks.org/maths/perimeter-formulas-for-geometric-shapes/)
> - [Quadrilateral](https://www.geeksforgeeks.org/maths/quadrilateral/)
> - [Perimeter of Triangle](https://www.geeksforgeeks.org/maths/perimeter-of-a-triangle/)
> - [Circumference of a Circle](https://www.geeksforgeeks.org/maths/circumference-of-circle/)

## Solved Examples on Perimeter of Square

Some Solved example problems on the Perimeter of ****a**** Square.

****Example 1: Find the perimeter of the square if the side given is 4 units.****   
****Solution:****

> The Perimeter of square is, P = 4 × side   
> ⇒ P = 4 × 4   
> ****⇒ P = 16 units****

****Example 2: Find the perimeter if the area given is 25 sq units.****  
****Solution:****

> Perimeter = 4√Area   
> ⇒ Perimeter = 4√25   
> ****⇒ Perimeter = 20 units****

****Example 3: Find the diagonal if the perimeter of the square is 3√2 cm.****  
****Solution:****

> Perimeter = 4 × (diagonal/√2)   
> ⇒ (3√2 × √2 ) /4 = diagonal  
> ****⇒ Diagonal = 1.5 cm****

****Example 4: Find the side and perimeter of the square if the diagonal given is 2√2 cm.****  
****Solution:****

> Side = Diagonal/√2   
> ⇒ Side = 2√2/√2   
> ⇒ Side = 2 cm
> 
> Perimeter of square = 4×side   
> ****⇒ Perimeter = 8 cm****


- [Perimeter of Square | Formula,](https://www.geeksforgeeks.org/maths/perimeter-of-square/) [Derivation, Examples](https://www.geeksforgeeks.org/maths/perimeter-of-square/)
- [Area and Perimeter of Shapes | Formula and Examples](https://www.geeksforgeeks.org/maths/area-and-perimeter/)
- [Square in Maths - Area, Perimeter, Examples & Applications](https://www.geeksforgeeks.org/maths/square/)


 an area if [the perimeter of a square](https://www.geeksforgeeks.org/maths/what-is-the-formula-for-perimeter-of-a-square/) is given.

Formula of the perimeter of a square = ****4 × side****

![Area of Square Using Perimeter](https://media.geeksforgeeks.org/wp-content/uploads/20230303122946/Area-of-Square-3.png)

From the above formula, we can find the side length by dividing the Perimeter by 4.

> ****Side length(s) = Perimeter/4****

Using side length we can find the area of the square by using the formula ****Area = side ×  side = (Perimeter/4)********2****.

****Example: Find**** the ****Area of the Square if the**** ****perimeter of a square is 36 cm.****

****Solution:****

> Given, perimeter = 36 cm
> 
> So, Side length=perimeter/4  
> Side(s) = 36/4 = 9 cm
> 
> From the side length we can calculate area of square by  
> Area = Side2 = 92 = 81 cm2  
> ****Area of square with perimeter 36 cm  is 81 cm********2********.****

****Related Articles:****

- [Area of Circle](https://www.geeksforgeeks.org/maths/area-of-a-circle/)
- [Area of Rectangle](https://www.geeksforgeeks.org/maths/area-of-rectangle/)
- [Area of Trapezium](https://www.geeksforgeeks.org/maths/area-of-trapezium/)

## Solved Examples on Area of Square

****Example 1: Find the Area of the Square if the**** ****perimeter of a square is 64cm.****

****Solution:****

> Given, perimeter = 64cm  
> So, Side length = perimeter/4
> 
> Side(s) = 64/4 = 16cm
> 
> From the side length we can calculate area of square by  
> Area =Side2  
> Area = 162 = 256 cm2
> 
> ****Area of square with perimeter 64cm  is 256cm********2********.****

****Example 2: Find the area of a square if the length of the diagonal is 12cm.****

****Solution:****

> Given, diagonal length (d) = 12 cm  
> Area = (1/2) × d2  
> Area = (1/2) × 122  
> Area = 144/2 = 72 cm2
> 
> ****Area of square is 72 cm********2********.****

****Example 3: The length of each side of a square is 5cm and the cost of painting it is Rs. 5 per sq. cm. Find total cost to paint the square.****

****Solution:****

> Given, Side length (s) = 5cm
> 
> Area of Square = s2  
> A = 52 = 25 cm2
> 
> For 1 sq.cm cost of painting is Rs 5.
> 
> ****Total Cost of painting the 25sq cm= 25 × 5 = Rs125****

****Example 4: A floor that is 60 m long and 30 m wide is to be covered by square tiles of side 6 m. Find the number of tiles required to cover the floor.****

****Solution:****

> Length of the floor = 60 m  
> Breadth of the floor = 30 m
> 
> Area of floor = length × breadth = 60 m × 30 m = 1800 sq. m  
> Length of one tile = 6 m  
> Area of one tile = side ×side = 6 m  × 6 m = 36 sq. m
> 
> No. of tiles required = (area of floor)/(area of one tile) = 1800/36  = 50 tiles
> 
> ****Total tiles required is 50****

****Example 5: What is the Area of a**** ****Square if the perimeter of a square is 24 cm?****

****Solution:****

> Given, perimeter = 24 cm
> 
> So, Side length = perimeter/4  
> Side(s) = 24/4 = 6cm
> 
> From the side length we can calculate area of square by  
> Area = Side2  
> Area = 62 = 36cm2
> 
> ****Area of square with perimeter 24 is 36cm********2********.****


- [****Area of square****](https://www.geeksforgeeks.org/maths/area-of-a-square/)
- [****Perimeter of square****](https://www.geeksforgeeks.org/maths/perimeter-of-square/)

## Diagonal of Square

Diagonals of the square are equal to a√2, where a is the side of the square. The length of both diagonals of a square is equal to each other. The relation between diagonals and sides of a square is given by [Pythagoras Theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/).

![Diagonal of Square](https://media.geeksforgeeks.org/wp-content/uploads/20221103122742/diagonalofsquare.png)

### Length of Diagonal of Square

The length of diagonal of a square is calculated using the [Pythagorean Theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/) as, Hypotenuse2 = Base2 + Perpendicular2

Hence,   
Diagonal2 = Side2 + Side2  
d2 = s2 + s2  
d2 = 2s2

> ****d = s√2****

### ****People Also Read:****

- [Quadrilateral](https://www.geeksforgeeks.org/maths/quadrilateral/)
- [Area of Rhombus](https://www.geeksforgeeks.org/maths/area-of-rhombus/)
- [Area of Triangle](https://www.geeksforgeeks.org/maths/area-of-triangle/)
- [Area of Square](https://www.geeksforgeeks.org/maths/area-of-a-square/)

## Applications of Squares in Real Life

- Floor tiles often come in square shapes, providing a uniform and tidy appearance.
- Many rooms in buildings have square or rectangular layouts, making them easier to design and construct.
- Squares are fundamental to geometry and algebra, used in various calculations and equations.
- Digital images are composed of square pixels arranged on a grid, forming pixel art.
- Screens of electronic devices like computer monitors and smartphones are often square or rectangular.
- Integrated circuits, the building blocks of electronic devices, are typically manufactured on square silicon wafers.
- Chess and checkers boards feature square grids, offering a strategic playing field.
- Many packages and storage containers are square or rectangular for efficient stacking and storage.
- Books generally have square or rectangular pages, making them easy to store on shelves and in boxes.
- Agricultural fields are often laid out in square or rectangular shapes for efficient land use and irrigation.
- Many sports fields and courts, including those for soccer, football, basketball, and tennis, have square or rectangular boundaries.
- Board games like Scrabble, Sudoku, and crossword puzzles often use square grids or boards.

![Squareexamples2](https://media.geeksforgeeks.org/wp-content/uploads/20221107163354/Squareexamples2.png)

Square Examples in Real-life

## Solved Examples on Squares

Some Examples of Square Formulas are,

****Example 1: A square has one of its sides measuring 24 cm. Calculate its area and perimeter.****  
****Solution:**** 

> Given,  
> Side of Square = 24 cm
> 
> Area of Square = a2  
> = 24 × 24 = 576 sq cm 
> 
> Perimeter of Square =  Sum of all sides of square = a + a + a + a = 4a   
> P = 4 × 24   
> P = 96 cm
> 
> Hence, area of square is 576 sq. cm and perimeter of square is 96 cm.

****Example 2: Find the area of a square park whose perimeter is 420 ft.****  
****Solution:****   

> Given,  
> Perimeter of Square Park = 420 ft  
> Perimeter of a Square = 4 × side  
> 4 × side = 420  
> Side = 420/4  
> Side = 105 ft
> 
> Formulae for Area of a Square = side2  
> Hence, Area of Square Park = (105)2   
> A = 105 × 105 = 11025 ft2
> 
> Thus, area of a square park whose perimeter is 420 ft is 11025 ft2
> 
and opposite sides parallel and all interior angles equal to 90° is called a Diagonals of squares bisect each other perpendicularly. Note that all [squares](https://www.geeksforgeeks.org/maths/square/) are rhombus but not vice-versa. 

![Diagram of Square](https://media.geeksforgeeks.org/wp-content/uploads/20220914105848/Square2-660x390.jpg)

#### Properties of Square

The properties of a square are:

- All four sides of a square are equal to each other.
- The interior angles of a square are 90°.
- The diagonal of a square bisects each other at 90°.
- The opposite sides are parallel, and the adjacent sides are [perpendicular](https://www.geeksforgeeks.org/maths/perpendicular-lines/) in a square.

|Square Formula|   |
|---|---|
|Area of Square|****side********2****|
|Perimeter of Square|****4 × side****|

Where side is the length of any one of the sides.

### Rectangle

Rectangle is a quadrilateral whose opposite sides are equal and parallel and all the interior angles equal to 90°.

Diagonals of a [rectangle](https://www.geeksforgeeks.org/maths/rectangle/) bisect each other.

![Rectangle Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20220914105825/Rectangle4-660x390.jpg)

Illustration of a Rectangle

> Note that all the rectangles are parallelograms, but the reverse of this is not true.

#### Rectangle Properties

These are some of the important properties of rectangle:

- The opposite sides of a rectangle are parallel and equal. In the above example, AB and CD are parallel and equal, and AC and BD are parallel and equal.
- All 4 angles of a rectangle are equal and are equal to 90°. ∠A = ∠B = ∠C = ∠D = 90°.
- The diagonals of a rectangle bisect each other and the diagonals of a rectangle are equal, here, AD = BC.

|Rectangle Formulas|   |
|---|---|
|Area of Rectangle|****length × width****|
|Perimeter of Rectangle|****2 × (length + width)****|

### Rhombus

Rhombus is a quadrilateral that has all sides equal and opposite sides parallel. Opposite angles of a rhombus are equal, and diagonals of the [Rhombus](https://www.geeksforgeeks.org/maths/rhombus/) bisect each other perpendicularly. .

![Rhombus Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20220906032532/Rhombus1-660x390.png)

Rhombus Diagram

> Note all rhombus are parallelograms, but the reverse of this is not true.

#### Properties of Rhombus

Here are some of the key properties of a Rhombus:

- All 4 sides of a rhombus are equal. AB = BC = CD = AD.
- The opposite sides of a rhombus are parallel and equal. In the image above, AB is parallel to CD and AD is parallel to BC.
- The diagonals of a rhombus Bisect each other at 90°.

|Rhombus Formulas|   |
|---|---|
|Area of Rhombus|****1/2 ​× (diagonal1 × diagonal2​)****|
|Perimeter of Rhombus|****4 × side****|

Where side is the length of any one of the sides.

### Parallelogram

Parallelogram is a quadrilateral whose opposite sides are equal and parallel. Opposite angles of a Parallelogram are equal, and its diagonals bisect each other.

![Parallelogram Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20220914110235/Parallelogram2-660x390.jpg)

Parallelogram Illustration

#### Properties of Parallelogram

The properties of a [Parallelogram](https://www.geeksforgeeks.org/maths/parallelogram/) are:

- The opposite sides of a parallelogram are parallel and equal. In the above example, AB and CD are parallel and equal, and AC and BD are parallel and equal.
- The opposite angles in a parallelogram are equal. ∠A = ∠D and ∠B = ∠C.
- The diagonals of a parallelogram bisect each other.

|Paralellogram Formulas|   |
|---|---|
|Area of Parallelogram|****base × height****|
|Perimeter of Parallelogram|****2 × (a+b)****|

Where, ****a**** and ****b**** are the adjacent sides of a parallelogram.

### Trapezium

A trapezium is a quadrilateral that has one pair of opposite sides parallel. In a regular trapezium, non-parallel sides are equal, and its base angles are equal.

The area of [trapezium](https://www.geeksforgeeks.org/maths/trapezium/) is 1/2 × Sum of parallel sides × Distance between them.

![Trapezium](https://media.geeksforgeeks.org/wp-content/uploads/20220906030249/Trapezium-660x390.png)

Trapezium Illustration

#### Properties of Trapezium

Here are two important properties of a trapezium:

- The sides of the trapezium that are parallel to each other are known as the bases of trapezium. In the above image, AB and CD are the base of the trapezium.
- The sides of the trapezium that are non-parallel are called the legs. In the above image, AD and BC are the legs.

|Trapezium Formulas|   |
|---|---|
|Area of Trapezium|****1/2 ​× (a+b) × (h)****|
|Perimeter of Trapezium|****a+b+c+d****|

Where a, b, c, d are the side of trapezium and (****a**** and ****b****) are the parallel sides and the height (****h****) is the perpendicular distance between these parallel sides.

### Kite

Kite has two pairs of equal adjacent sides and one pair of opposite angles equal. Diagonals of [kites](https://www.geeksforgeeks.org/maths/kite-quadrilaterals/) intersect perpendicularly.

The longest diagonal of the kite bisects the smaller one.

![Kite Diagram](https://media.geeksforgeeks.org/wp-content/uploads/20220906030641/Kite-660x390.png)

Kite Illustration

#### Properties of Kite

Let's discuss some of the properties of a kite.

- A kite has two pairs of equal adjacent sides. For example, AC = BC and AD = BD.
- The interior opposite angles that are obtuse are equal; here, ∠A = ∠B.
- The diagonals of a kite are perpendicular to each other; here, AB is perpendicular to CD.
- The longer diagonal of the kite bisects the shorter diagonal. Here, CD bisects AB.

|Kite Formulas|   |
|---|---|
|Area of Kite|****½ (diagonal1 x diagonal2)****|
|Perimeter of Kite|****2(a + b)****|

where, ****a**** and ****b**** represent the lengths of the ****two pairs of equal sides**** of the kite.

## Quadrilateral Theorems

- ****Sum of Interior Angles Theorem****: In any quadrilateral, the sum of the measures of its interior angles equals 360 degrees.
- ****Opposite Angles Theorem****: Within a quadrilateral, the sum of the measures of two opposite angles is 180 degrees.
- ****Consecutive Angles Theorem****: Adjacent (consecutive) angles in a quadrilateral are supplementary, meaning their measures sum up to 180 degrees.
- ****Diagonals of Parallelograms Theorem****: The diagonals of a parallelogram bisect each other, dividing each diagonal into two equal segments.
- ****Opposite Sides and Angles of Parallelograms Theorem****: In a parallelogram, opposite sides are equal in length, and opposite angles are congruent.
- ****Diagonals of Rectangles and Rhombuses Theorem****: In rectangles and rhombuses, the diagonals are equal in length. Additionally, the diagonals of a rectangle are congruent, while those of a rhombus bisect each other at right angles.
- ****Diagonals of Trapezoids Theorem****: The diagonals of a trapezoid may have different lengths. However, the segment joining the midpoints of the non-parallel sides is parallel to the bases and is equal to half their sum.

## Quadrilateral Lines of Symmetry

A quadrilateral has lines of symmetry that are imaginary lines that pass through the center of the quadrilateral and divide it into two similar halves. A line of symmetry can:

- Match two vertices on one side of the line with two vertices on the other.
- Pass through two vertices, and the other two vertices pair up when folded over the line.

A regular quadrilateral has four lines of symmetry. For example, a square has four lines of symmetry, including both its diagonals and the lines joining the midpoints of its opposite sides. A rectangle has two lines of symmetry, including the lines joining the midpoint of the opposite and parallel lines.

## Quadrilateral Sides and Angles

The following table illustrates how the sides and angles of quadrilaterals make them different from one another:

|****Characteristics of Quadrilaterals****|   |   |   |   |   |
|---|---|---|---|---|---|
|****Sides and angles****|****Square****|****Rectangle****|****Rhombus****|****Parallelogram****|****Trapezium/Trapezoid****|
|---|---|---|---|---|---|
|All sides are equal|Yes|No|Yes|No|No|
|Opposite sides are parallel|Yes|Yes|Yes|Yes|Yes ( Only one pair of opposite sides are parallels)|
|Opposite sides are equal|Yes|Yes|Yes|Yes|No|
|All the angles are of the same measure|Yes (90°)|Yes (90°)|No|No|No|
|Opposite angles are of equal measure|Yes|Yes|Yes|Yes|No|
|Diagonals bisect each other|Yes|Yes|Yes|Yes|No|
|Two adjacent angles are supplementary|Yes|Yes|Yes|Yes|Yes ( Only adjacent Angles along the non parallel side are supplementary)|
|Lines of Symmetry|4|2|2|0|0|

## Solved Examples on Quadrilaterals

Here are some solved examples on quadrilaterals for your help.

****Question 1: The perimeter of quadrilateral ABCD is 46 units. AB = x + 7, BC = 2x + 3, CD = 3x - 8, and DA = 4x - 6. Find the length of the shortest side of the quadrilateral.**** 

****Solution****:

> Perimeter = Sum of all sides
> 
> = 46 = 10x - 4 or [x = 5]
> 
> That gives, AB = 12 units, BC = 13 units, CD = 7 units, DC = 14 units
> 
> Hence, l****ength of shortest side is 7 units (i.e. CD).****

****Question 2: Given a trapezoid ABCD (AB || DC) with median EF. AB = 3x - 5, CD = 2x -1 and EF = 2x + 1. Find the value of EF.****

****Solution****:

> We know that the Median of the trapezoid is half the sum of its bases.
> 
> = EF = (AB + CD) / 2
> 
> = 4x + 2 = 5x - 6  or [x = 8]
> 
> Therefore EF = 2x + 1 = 2(8) + 1 => EF = 17 units.

****Question 3: In a Parallelogram, adjacent angles are in the ratio of 1:2. Find the measures of all angles of this Parallelogram.****

****Solution:****

> Let the adjacent angle be x and 2x.
> 
> We know that in of a Parallelogram adjacent angles are supplementary.
> 
> ![Quadrilateral Solved Example](https://media.geeksforgeeks.org/wp-content/uploads/20220906033837/Quadrilateralsolvedexample3-660x390.png)
> 
> = x + 2x = 180° or [x = 60°]
> 
> Also, opposite angles are equal in a Parallelogram.
> 
> Therefore measures of each angles are ****60°, 120°, 60°, 120°.****

|Articles related to Quadrilateral|   |
|---|---|
|[Types of Polygons](https://www.geeksforgeeks.org/maths/types-of-polygons/)|[Area of a Quadrilateral](https://www.geeksforgeeks.org/maths/area-of-quadrilateral/)|
|[Construction of a Quadrilateral](https://www.geeksforgeeks.org/maths/construction-of-a-quadrilateral/)|[Area of Cyclic Quadrilateral](https://www.geeksforgeeks.org/maths/cyclic-quadrilateral/)|

### Quadrilateral Notes and Solution For Class 9

> - [Quadrilateral Class 9 Notes](https://www.geeksforgeeks.org/maths/cbse-class-9-maths-notes/#:~:text=Chapter%C2%A08%3A%20Quadrilateral%20%C2%A0%C2%A0)
> - [Quadrilateral Class 9 NCERT Solutions](https://www.geeksforgeeks.org/maths/ncert-solutions-for-class-9-maths/#:~:text=Chapter%207%20%E2%80%93%20Triangles-,Chapter%C2%A08%20%E2%80%93%20Quadrilateral,-The%20chapter%20Quadrilateral)

## Conclusion

Quadrilateral is a two-dimensional shape with four sides, corners, and angles, with a total interior angle sum of 360 degrees. There are two main types: concave, which has at least one angle greater than 180 degrees, and convex, where all angles are less than 180 degrees. Convex quadrilaterals include trapezoids, parallelograms, rectangles, rhombuses, squares, and kites. Each type has unique properties and formulas for calculating area and perimeter. For instance, the area of a rectangle is length times width, while a rhombus’s area is half the product of its diagonals. Symmetry and properties like equal sides or right angles vary among these shapes, making each useful for different applications in fields like architecture and design.

- [Quadrilateral](https://www.geeksforgeeks.org/maths/quadrilateral/)
- [Properties of parallelogram](https://www.geeksforgeeks.org/maths/properties-of-parallelograms/)

## Types of Parallelogram

There are mainly four types of parallelograms, based on their properties.

- Rectangle
- Square
- Rhombus
- Rhomboid

### Rectangle

A rectangle is a parallelogram with two pairs of equal and parallel opposite sides, along with four right angles.

Observe the rectangle ABCD and associate it with the following properties,

1. Two pairs of parallel sides. Here AB || DC and AD || BC
2. Four right angles ∠A = ∠B = ∠C = ∠D = 90°.
3. The opposite sides are the same length, where AB = DC and AD = BC.
4. Two equal diagonals where AC = BD.
5. Diagonals that bisect each other.

Read in Detail: [****Rectangle****](https://www.geeksforgeeks.org/maths/rectangle-formula/)

![Rectangle Definition](https://media.geeksforgeeks.org/wp-content/uploads/20230416203040/Parallelogram-7.png)

Diagram of a Rectangle

### Square

A square is a parallelogram with four equal sides and four equal angles.

Observe the square ACDB and associate it with the following properties:

1. Four equal sides are AB = BC = CD = DA.
2. Right angles are ∠A = ∠B = ∠C = ∠D = 90°.
3. There are two pairs of parallel sides. Here AB || DC and AD || BC.
4. Two identical diagonals where AD = BC.
5. Diagonals are perpendicular and bisect each other; AD is perpendicular to BC.

****Read in Detail:**** The perimeter[****Square****](https://www.geeksforgeeks.org/maths/square/)****.****

![Square Definition](https://media.geeksforgeeks.org/wp-content/uploads/20230416203111/Parallelogram-1-(1).png)

Diagram of a Square

### Rhombus

A parallelogram with four equal sides and equal opposite angles is called a rhombus. Consider the diamond ABCD and assign it the following attributes, 

1. In the given figure, the four equal sides are AB = CD = BC = AD. 
2. The two pairs of parallel sides are AB || CD and BC || AD. 
3. The equal opposite angles are ∠A = ∠B and ∠C = ∠D. 
4. Here, the diagonals (AC and BD) are perpendicular to each other and bisect at right angles.

****Read in Detail:**** [****Rhombus****](https://www.geeksforgeeks.org/maths/rhombus/)****.****

![Rhombus Definition](https://media.geeksforgeeks.org/wp-content/uploads/20230416203105/Parallelogram-2-(1).png)

Diagram of a Rhombus

### Rhomboid

A ****rhomboid**** is a quadrilateral with opposite sides that are parallel and equal in length, but angles are not necessarily right angles. Consider the rhomboid ABCD with the following attributes:

- The equal opposite sides are AB = CD and BC = AD.
- The two pairs of parallel sides are AB ∥ CD and BC ∥ AD.
- The opposite angles are equal: ∠A = ∠C and ∠B = ∠D.
- The diagonals (AC and BD) bisect each other but are not perpendicular.

****Read in Detail:**** [****Rhomboid****](https://www.geeksforgeeks.org/maths/rhomboid/)

![Rhomboid](https://media.geeksforgeeks.org/wp-content/uploads/20240926133330/Rhomboid.webp)

RHomboid

## Parallelogram Formulas | Area and Perimeter

All 2D shapes have two basic formulas for area and perimeter. Parallelogram is a basic 2-dimensional figure which is widely used in mathematics.

All the formulas on parallelogram can be subdivided into two parts:

- Area
- Perimeter

### Area of Parallelogram

The area of ​​a parallelogram is the space covered between its four sides. It can be calculated by knowing the length of the base and the height of the parallelogram and measuring it in square units such as cm2, m2, or inch2.

Consider a parallelogram ABCD with a base (b) and a height (h).  
Then, the [area of ​​a parallelogram](https://www.geeksforgeeks.org/maths/area-of-parallelogram/) is calculated by the formula:

> ****Area of Parallelogram = base (b) × height (h)****

### Area of Parallelogram without Height

When the [height of the parallelogram](https://www.geeksforgeeks.org/maths/height-of-a-parallelogram-formula/) is not known, the area can still be found, provided the angle is known to us.

The formula for the area of a parallelogram without height is given as:

> ****Parallelogram area = ab Sinθ****

Where a and b are the sides of the parallelogram and θ is the angle between them.

### The perimeter of Parallelogram

[Perimeter of a parallelogram](https://www.geeksforgeeks.org/maths/perimeter-of-a-parallelogram/) is the length of its boundary, so it is equal to the sum of all sides.

In a parallelogram, the opposite sides are equal. Let's say the sides are a and b. Then, the perimeter (P) of the parallelogram with edges is in units of P = 2 (a + b).

> ****Perimeter of Parallelogram = 2 (a + b)****

## Formulas Table

|****Property/Formula****|****Formula/Description****|
|---|---|
|****Area (A)****|A = b × h|
|****Perimeter (P)****|P = 2(a + b)|
|****Base (b)****|Length of the base side.|
|****Height (h)****|Length of the perpendicular height from base to opposite side.|
|****Length of Sides****|a, ba, ba,b are the lengths of the adjacent sides.|
|****Diagonal Lengths****|d1 = a2+b2+2ab cos(θ)a2+b2+2ab cos(θ)​<br><br>d2 = a2+b2−2ab cos(θ)a2+b2−2ab cos(θ)​|
|****Angles****|Opposite angles are equal: ∠A = ∠C  <br>Adjacent angles are supplementary: ∠A + ∠B = 180∘|
|****Relationship Between Sides and Angles****|a sin⁡(∠B) = b sin⁡(∠A)|
|****Area using diagonals****|A=12×d1×d2×sin⁡(θ)A=21​×d1​×d2​×sin(θ)|
|****Circumradius (R)****|R = d1×d22A2Ad1​×d2​​|
|****Inradius (r)****|r = A/P|

## Parallelogram Theorem

Let's understand the theorem on parallelograms and how to prove it.

****Theorem:Parallelograms on the same base and between the same parallels are equal in area.****

> ****To Prove:**** Area of parallelogram ABCD = Area of parallelogram ABEF
> 
> ****Proof:**** Let's assume two parallelograms ABCD and ABEF with the same base DC and between the same parallel lines AB and FC
> 
> In the figure given below, the two parallelograms, ABCD and ABEF, lie between the same parallel lines and have the same base. Area ABDE is common between them.
> 
> Taking a closer look at the two triangles, △BCD and △AEF might be congruent. 

![Parallelogram Theorem](https://media.geeksforgeeks.org/wp-content/uploads/20230416203245/Parallelogram-5-(1).png)

Parallelogram Theorem

> BC = AE (Opposite sides of a parallelogram), 
> 
> ∠BCD = ∠AEF (These are corresponding angles because BC || AE and CE are the transversal).
> 
> ∠BDC = ∠AFE (These are corresponding angles because BD || AF and FD are the transversals).
> 
> Thus, by the ASA criterion of congruent triangles. These two triangles are congruent, and they must have equal areas. 
> 
> area(BCD) = area(AEF)
> 
> area(BCD) + area(ABDE) = area(AEF) + area(ABDE) 
> 
> area(ABCD) = area(ABEF) 
> 
> Hence, parallelograms lying between the same parallel lines and having a common base have equal areas. 

## Difference Between Parallelogram and Rectangle

Rectangle and parallelogram are both quadrilaterals. All rectangles are parallelograms as they have all the properties of a parallelogram and more but all parallelograms are not rectangles.

Here we have tabulated some of the basic differences between their properties:

|Properties|Parallelogram|Rectangle|
|---|---|---|
|Sides|The opposite sides of a parallelogram are equal.|The opposite sides of a rectangle are equal.|
|Diagonals|The diagonals of a parallelogram bisect each other, but the diagonals are not equal.|The diagonals of a rectangle bisect each other, and the diagonals are equal to each other as well.|
|Angles|The opposite angles of a parallelogram are equal, and the adjacent angles are supplementary.|All the angles of a rectangle are equal to each other and equal to 90°.|

****Also Check,****

> - [Area of a Triangle](https://www.geeksforgeeks.org/maths/area-of-triangle/)
> - [Area of a Square](https://www.geeksforgeeks.org/maths/area-of-a-square/)
> - [Area of Rectangle](https://www.geeksforgeeks.org/maths/area-of-rectangle/)

## Real-life examples of a Parallelogram

Various examples of parallelograms as observed in our daily life include:

- We come across various things in our daily life that resembles a parallelogram such as a computer screen, books, buildings, and tiles all are considered to be in a parallelogram shape.
- The parallelogram is the most common shape which we encounter daily.
- Rectangle and square both can be considered a parallelogram and are easily seen in our daily lives.

## Solved Examples on Parallelogram

****Example 1: Find the length of the other side of a parallelogram with a base of 12 cm and a perimeter of 60 cm.****  
****Solution:****

> Given perimeter of a parallelogram = 60cm.  
> Base length of given parallelogram = 12 cm.   
> P = 2 (a + b) units 
> 
> Where b = 12cm and P = 40cm.  
> 60 = 2 (a + 12)  
> 60 = 2a + 24  
> 2a = 60 - 24  
> 2a = 36  
> a = 18cm
> 
> Therefore, the length of the other side of the parallelogram is 18 cm. 

****Example 2: Find the perimeter of a parallelogram with the base and side lengths of 15cm and 5cm, respectively.****  
****Solution:****

> Base length of given parallelogram = 15 cm  
> Side length of given parallelogram = 5 cm
> 
> Perimeter of a parallelogram is given by,  
> P = 2(a + b) units.
> 
> Putting the values, we get  
> P = 2(15 + 5)  
> P = 2(20)  
> P = 40 cm
> 
> Therefore, the perimeter of a parallelogram will be 40 cm.

****Example 3: The angle between two sides of a parallelogram is 90°. If the lengths of two parallel sides are 5 cm and 4 cm, respectively, find the area.****

![Parallelogram Solved Example](https://media.geeksforgeeks.org/wp-content/uploads/20230416203032/Parallelogram-8.png)

****Solution:****

> If one angle of the parallelogram is 90°. Then, the rest of the angles are also 90°. Therefore, the parallelogram becomes a rectangle. The area of the rectangle is length times breadth.  
> Area of parallelogram = 5 × 4  
> Area of parallelogram = 20cm2

****Example 4: Find the area of a parallelogram when the diagonals are given as 8 cm, and 10 cm, the angle between the diagonals is 60°.****  
****Solution:****

> In order to find the area of the parallelogram, the base and height should be known, lets's first find the base of the parallelogram, applying the law of cosines,
> 
> b2 = 42 + 52 - 2(5)(4)cos(120°)  
> b2 = 16 + 25 - 40(0.8)  
> b2 = 9  
> b = 3cm 
> 
> Finding the height of the parallelogram,
> 
> ![Parallelogram Solved Example](https://media.geeksforgeeks.org/wp-content/uploads/20230416203059/Parallelogram-3-(1).png)
> 
> 4/sinθ = b/sin120  
> 4/sinθ = 3/-0.58  
> sinθ = -0.773  
> θ = 50°
> 
> Now, to find the height,
> 
> Sinθ = h/10  
> 0.76 = h/10  
> h = 7.6cm
> 
> Area of the parallelogram = 1/2 × 3 × 7.6 = 11.4 cm2

In essence, an oblique triangular prism doesn't have the strict right-angle alignment between its triangular ends and its rectangular sides. Instead, the lateral faces form [parallelograms](https://www.geeksforgeeks.org/maths/parallelogram/), allowing for more flexibility in the geometric configuration of the prism.

### Other Types of Prism

- [Rectangular Prism](https://www.geeksforgeeks.org/maths/rectangular-prism/)
- [Pentagonal Prism](https://www.geeksforgeeks.org/maths/pentagonal-prism/)
- [Square Prism](https://www.geeksforgeeks.org/maths/square-prism/)
- [Hexagonal Prism](https://www.geeksforgeeks.org/maths/hexagonal-prism/)

## Properties of Triangular Prism

|Parts of a Triangular Prism|Numbers|
|---|---|
|Face of a Triangular Prism|5|
|Edge of a Triangular Prism|9|
|Vertex of a Triangular Prism|6|

A triangular prism is easily identifiable by its key characteristics. Here are the important properties explained in neutral language:

- ****Polyhedral Nature:**** It falls under the category of Polyhedra, that specifically characterized by two triangular bases and three rectangular sides.
- ****Base Shape:**** The base of a triangular prism is in the shape of a triangle.
- ****Side Shape:**** The sides are in the shape of rectangles, providing a consistent structure along the length of the prism.
- ****Equilateral Triangular Bases:**** The two triangular bases are equilateral triangles, meaning all sides of these triangles are of equal length.
- ****Cross-Section Shape:**** Any cross-section of triangular prism results in the shape of a triangle.
- ****Congruent Bases:**** The two triangular bases are identical to each other that implies their congruence

## Triangular Prism Net

The net of a triangular prism is like a blueprint that unfolds the surface of the prism. By folding this net, you can recreate the original triangular prism.

The net illustrates that the prism has triangular bases and rectangular lateral faces. In simpler terms, it's a visual guide that shows how the prism can be assembled from a flat, folded shape.

![Triangular Prism-Net](https://media.geeksforgeeks.org/wp-content/uploads/20231124185525/Prism-Net.png)

Triangular Prism Net

## Surface Area of a Triangular Prism

The Surface Area of a Triangular Prism is divided into two parts [Lateral Surface Area](https://www.geeksforgeeks.org/maths/lateral-area-formula/) and Total Surface Area

### Lateral Surface Area (LSA) of a Triangular Prism:

The lateral surface area (LSA) of a triangular prism is the total area of all its sides excluding the top and bottom faces. The formula to calculate the lateral surface area is given by:

Lateral Surface Area (LSA) = (s1 + s2 + h)L

Here, s1, s2, and s3 are the lengths of the edges of the base triangle, and L is the length of the prism.

For a right triangular prism, the formula is:

> ****Lateral Surface Area = (s********1**** ****+ s********2**** ****+ h)L****
> 
> ****OR****
> 
> ****Lateral Surface Area = Perimeter × Length****

Here, (h) represents the height of the base triangle, (L) is the length of the prism, and s1 and s2 are the two edges of the base triangle.

### Total Surface Area (TSA) of a Triangular Prism

The total surface area (TSA) of a triangular prism is found by adding the area of its lateral surface (the sides) and twice the area of one of its triangular bases. For a right triangular prism, where one of the bases is a right-angled triangle, the formula for the total surface area is given by:

> ****Total Surface Area (TSA) = (b × h) + (s********1**** ****+ s********2**** ****+ s********3********) L****

Here, s1, s2, and s3 are the edges of the triangular base, (h) is the height of the base triangle, (l) is the length of the prism, and (b) is the bottom edge of the base triangle.

For a right triangular prism specifically, the formula simplifies to:

> ****Total Surface Area = (s********1**** ****+ s********2**** ****+ h) L + b × h****

Where,

- _****b****_ is the bottom edge of the base triangle.
- _****h****_ is the height of the base triangle.
- _****L****_ is the length of the prism.
- _****s****__****1****_ and _****s****__****2****_ represent the two edges of the base triangle.
- _****bh****_ represents the combined area of the two triangular faces.
- _****(s****__****1****_ _****+ s****__****2****_ _****+ h) L****_ represents the combined area of the three rectangular side faces.

This formula essentially accounts for the areas of all the faces (rectangular and triangular) of the prism, providing a comprehensive measure of its total surface area.

## Volume of Triangular Prism

The volume of triangular prism refers to the amount of space it occupies in the three-dimensional space. The formula to compute the volume of triangular prism is expressed as:

> ****Volume (V) = 1/2 × base edge × height of the triangle × length of the prism****

Where,

- base edge _****b****_: This is the length of one of the edges forming the base triangle.
- height of the triangle _****h****_: It represents the perpendicular distance from the base to the opposite vertex, forming the triangle.
- length of the prism _****l****_: This indicates the overall length of the prism along its axis.

By using these values in the formula, one can calculate the volume of the triangular prism.

****Also Read:****

- [Volume of Triangular Prism](https://www.geeksforgeeks.org/maths/volume-of-a-triangular-prism-formula/)
- [Volume of square Prism](https://www.geeksforgeeks.org/maths/volume-of-a-square-prism/)
- [Volume of Rectangular prism](https://www.geeksforgeeks.org/maths/volume-of-a-rectangular-prism-formula/)

## Solved Examples on Triangular Prism

****Example 1. Consider a triangular prism with a base edge of 4 cm, a height of the triangular base as 6 cm, and an overall length of the prism as 10 cm. Find the volume of triangular prism.****

****Solution:****

> Given:
> 
> - Base edge (b) = 4 cm
> - Height of the triangular base (h) = 6 cm
> - Length of the prism (l) = 10 cm
> 
> The formula for the volume (V) of a triangular prism is:
> 
> V= 1/2 × b × h × l
> 
> Substitute the given values into the formula:
> 
> V= 1/2 × 4cm × 6cm × 10cm
> 
> V=120cm3
> 
> ∴ the volume of the triangular prism is 120cm3

****Example 2. A triangular prism has a triangular base with sides measuring 8 cm, 15 cm, and 17 cm. The height of the triangular base is 10 cm, and the overall length of the prism is 12 cm. Calculate the surface area of triangular prism.****

****Solution:****

> Given:
> 
> - Sides of the triangular base (a, b, c) = 8 cm, 15 cm, 17 cm (this is a right-angled triangle)
> - Height of the triangular base (h) = 10 cm
> - Length of the prism (l) = 12 cm
> 
> The formula for the surface area (A) of a triangular prism is:
> 
> A=2 × area of base triangle + perimeter of base × height of prism
> 
> First, calculate the area of the base triangle using Heron's formula:
> 
> s= (a + b + c)/2
> 
> Area= √[s × (s - a) × (s - b) × (s - c)]
> 
> s = (8 + 15 + 17) / 2 = 20
> 
> Area= √20 × (20 - 8) × (20 - 15) × (20 - 17)
> 
> Area= √20 × 12 × 5 × 3
> 
> = √3600
> 
> = 60cm2
> 
> Now, substitute the values into the surface area formula:
> 
> A= 2 × 60cm2 +(8+15+17)cm × 10cm
> 
> A= 120cm2 + 40cm × 10cm
> 
> A= 520cm2
> 
> ∴ the surface area of the triangular prism is 520 cm2

****Example 3. Consider a triangular prism with a base edge of 9 cm, a height of the triangular base as 16 cm, and an overall length of the prism as 20 cm. Find the volume of triangular prism.****

****Solution:****

> Given:
> 
> - Base edge (b) = 9 cm
> - Height of the triangular base (h) = 16 cm
> - Length of the prism (l) = 20 cm
> 
> The formula for the volume (V) of a triangular prism is:
> 
> V= 1/2 × b × h × l
> 
> Substitute the given values into the formula:
> 
> V= 1/2 × 9 cm × 16 cm × 20cm
> 
> V= 1440 cm3
> 
> ∴ the volume of the triangular prism is 1440 cm3

## Triangular Prism - Practice Questions

****Question 1****. A triangular prism has a triangular base with sides measuring 10 cm, 18 cm, and 25 cm. The height of the triangular base is 12 cm, and the overall length of the prism is 15 cm. Calculate the total surface area of triangular prism.

****Question 2.**** Consider a triangular prism with a base edge of 8 cm, a height of the triangular base as 10 cm, and an overall length of the prism as 16 cm. Find the volume of triangular prism.

****Question 3****. A triangular prism has a triangular base with sides measuring 5 cm, 9 cm, and 13 cm. The height of the triangular base is 15 cm, and the overall length of the prism is 25 cm. Calculate the lateral surface area of triangular prism.

****Question 4.**** Consider a triangular prism with a base edge of 9 cm, a height of the triangular base as 17 cm, and an overall length of the prism as 30 cm. Find the volume of triangular prism.

## Conclusion

Triangular prism is a three-dimensional shape with two triangular bases and three rectangular sides. Triangular prisms come in different types, such as right, oblique, and regular, each with unique characteristics and properties. Triangular prisms have various practical uses in both everyday life and specialized fields:

- ****Architecture****: Triangular prisms are used in building designs, particularly for roofs, bridges, and support structures, providing stability and a unique shape.
- ****Optics****: Glass triangular prisms are used to split or bend light in optics and science experiments, such as in spectrometers and lasers.
- ****Tents and Shelters****: Many tents are shaped like triangular prisms because of their simple and stable structure.
- ****Aquariums****: Triangular prism-shaped tanks can be found in certain types of aquariums for unique visual effects.
- ****Mathematics and Education****: Triangular prisms are used in geometry lessons to help students understand 3D shapes and their properties, such as volume and surface area.

- [****Triangular Prism****](https://www.geeksforgeeks.org/maths/triangular-prism/)
- [****Hexagonal Prism****](https://www.geeksforgeeks.org/maths/hexagonal-prism/)
- [****Octagonal Prism****](https://www.geeksforgeeks.org/maths/octagonal-prism/)
- [****Pentagonal Pyramid****](https://www.geeksforgeeks.org/maths/pentagonal-pyramid/)
- [****Rectangular Pyramid****](https://www.geeksforgeeks.org/maths/rectangular-pyramid/)
- [****Regular Tetrahedron****](https://www.geeksforgeeks.org/maths/regular-tetrahedron-formula/)


| Polyhedrons                                                       | Characteristics                                                                                                                | Shape or Form                                                                                        |                                     |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------- |
|                                                                   | - ****Polyhedron Composed of Squares****<br>- ****Faces: 6****<br>- ****Vertices: 8****<br>- ****Edges: 12****                 | ![Cube](https://media.geeksforgeeks.org/wp-content/uploads/20230703171020/Polyhedrons-1.png)         |                                     |
| Tetrahedron                                                       | - ****Polyhedron Composed of Equilateral Triangles****<br>- ****Faces: 4****<br>- ****Vertices: 4****<br>- ****Edges: 6****    | ![Tetrahedron](https://media.geeksforgeeks.org/wp-content/uploads/20230703171020/Polyhedrons-2.png)  |                                     |
| Octahedron                                                        | - ****Polyhedron Composed of Equilateral Triangles****<br>- ****Faces: 8****<br>- ****Vertices: 6****<br>- ****Edges: 12****   | ![Octahedron](https://media.geeksforgeeks.org/wp-content/uploads/20230703171021/Polyhedrons-3.png)   |                                     |
| [Dodecahedron](https://www.geeksforgeeks.org/maths/dodecahedron/) | - ****Polyhedron Composed of Regular Pentagons****<br>- ****Faces: 12****<br>- ****Vertices: 20****<br>- ****Edges: 30****     | ![Dodecahedron](https://media.geeksforgeeks.org/wp-content/uploads/20230703171022/Polyhedrons-5.png) |                                     |
| Icosahedron                                                       | - ****Polyhedron Composed of Equilateral Triangles****<br>- ****Faces: 20****<br>- ****Vertices: 12****<br>- ****Edges: 30**** | ![Icosahedron](https://media.geeksforgeeks.org/wp-content/uploads/20230703171021/Polyhedrons-4.png)  | # Real-Life Examples of Polyhedrons |

The following illustration contains some real-life examples of polyhedrons:

![Real-Life Examples](https://media.geeksforgeeks.org/wp-content/uploads/20230703171025/Polyhedrons-9.png)

## Polyhedrons Faces, Edges, and Vertices

- ****Faces****: The flat, two-dimensional polygons that make up the polyhedron's surface are known as faces.
- ****Edges****: The edges of a polyhedron are the segments of a straight line that connect two faces. They define the boundaries or points where the faces converge.
- ****Vertices****: Vertices are the polyhedron's corners or meeting points for multiple edges.

![Polyhedron: Faces, Edges & Vertices](https://media.geeksforgeeks.org/wp-content/uploads/20230703171022/Polyhedrons-6.png)

****Read More:**** [****Vertices, Faces, and Edges****](https://www.geeksforgeeks.org/maths/faces-edges-and-vertices/)****.****

## Prisms, Pyramids, and Platonic Solids

![Prisms & Pyramids](https://media.geeksforgeeks.org/wp-content/uploads/20230707161044/Polyhedron-2-(1).png)

### Prisms

Prisms are polyhedrons with two parallelogram-shaped lateral faces connecting two congruent polygonal bases. They can be found as triangular, rectangular, or pentagonal prisms, among other shapes. Prisms are frequently found in commonplace items like buildings and packaging.

- [****Triangular Prism****](https://www.geeksforgeeks.org/maths/triangular-prism/)****:**** It has triangular bases and three rectangular ****lateral faces****( faces of a polyhedron that are not based).
- [****Rectangular Prism****](https://www.geeksforgeeks.org/maths/rectangular-prism/)****:**** It has rectangular bases and four rectangular lateral faces.
- [****Pentagonal Prism****](https://www.geeksforgeeks.org/maths/pentagonal-prism/)****:**** It has pentagonal bases and five rectangular lateral faces.

### Pyramids

Pyramids are polyhedrons with triangular faces that converge at a single vertex known as the apex along with a polygonal base. Tetrahedrons, square pyramids, and pentagonal pyramids are a few examples of pyramid shapes. Pyramids have been used in construction, including the Egyptian pyramids, and are frequently related to past civilizations.

- ****Tetrahedron:**** It has three triangle faces that converge at the top.
- ****Square Pyramid:**** Four triangular faces that converge at the top and have a square base.
- [****Pentagonal Pyramid****](https://www.geeksforgeeks.org/maths/pentagonal-pyramid/)****:**** This structure has five triangular faces that converge into a pentagonal base.

### Platonic Solids

Five convex polyhedrons with identical regular polygonal faces and equal angles make up a distinctive category called "Platonic solids." They consist of the cube, octahedron, dodecahedron, and icosahedron, as well as the tetrahedron. Mathematicians and philosophers have been attracted to the unique symmetry characteristics of platonic solids for centuries. They are related to the philosophical elements of Plato and are seen as depicted geometric forms.

### ****People Also Read:****

> - [Platonic Solids](https://www.geeksforgeeks.org/maths/what-are-platonic-solids/)
> - [Platonic Solid Formula](https://www.geeksforgeeks.org/maths/platonic-solids-formula/)
> - [Prism Formula](https://www.geeksforgeeks.org/maths/prism-formula/)

## Polyhedron Types

Polyhedrons can be classified into various categories, based on various parameters.

- Based on Edge Length 
    - Regular Polyhedron
    - Irregular Polyhedron

- Based on the Surface Diagonal
    - Convex Polyhedron
    - Concave Polyhedron

Let's understand these types in detail as follows:

### Regular Polyhedron

A regular polyhedron is one whose edges are of the same length and is made up of regular polygons. It is a three-dimensional object with sharp vertices and flat faces made of straight edges. These polyhedrons are commonly known as Platonic solids.  
The arrangement of vertices, edges, and faces in regular polyhedrons demonstrates symmetry, and the faces are congruent regular polygons.

Some common examples of regular polyhedrons are tetrahedrons, cubes, octahedrons, dodecahedrons, and icosahedrons.

![Regular Polyhedron](https://media.geeksforgeeks.org/wp-content/uploads/20230703171023/Polyhedrons-7.png)

### Irregular Polyhedron

Polyhedrons that don't fit into the criteria of regularity are called irregular polyhedrons. Their vertices, edges, and faces are not symmetrically arranged, and they do not all have congruent or regular polygonal faces.  
Irregular polyhedrons can have faces of various sizes and forms, as well as variable edge and vertices combinations.

Some common examples of irregular polyhedrons are Cuboid, Irregular Dodecahedrons, and Irregular Icosahedrons.

![Irregular polyhedron](https://media.geeksforgeeks.org/wp-content/uploads/20230707161043/Irregular-Polyhedron.png)

### Convex Polyhedron

Every line segment joining any two points inside the polyhedron completely resides inside or on the polyhedron's surface in a convex polyhedron. In other terms, it is a polyhedron with convex polygons on each face and flat surfaces throughout.

****Properties of Convex Polyhedron:****

- All of its faces' inner angles are less than 180 degrees.
- Any two faces' intersections are either empty, share an edge, or have a common vertex.

Examples: regular tetrahedron, cube, octahedron, dodecahedron, icosahedron, etc.

  

![Convex Polyhedron](https://media.geeksforgeeks.org/wp-content/uploads/20230703171024/Polyhedrons-8.png)

  

### Concave Polyhedron

A concave polyhedron is a particular kind of polyhedron that has at least one concave face, or one with an interior angle higher than 180 degrees.

There are line segments connecting points inside a concave polyhedron that may extend beyond the polyhedron's surface. This indicates that in some areas of the polyhedron, the line segment joining two points does not wholly lie inside or on the polyhedron's surface.

****Examples:**** star-shaped polyhedron, Stair-Case-shaped polyhedron.

![Concave Polyhedron](https://media.geeksforgeeks.org/wp-content/uploads/20230703171323/Polyhedrons-10-.png)

### Some Other Types of Polyhedrons

- ****Archimedean Solids:**** Archimedean solids are those convex polyhedrons that have equal edges but have different types of regular polygonal faces. Some examples of these solids include the truncated icosahedron (soccer ball shape) and the rhombicuboctahedron.

- ****Johnson Solids:**** Johnson solids are convex polyhedrons that are not regular or Archimedean. They have faces that are regular polygons, but the arrangement of the faces and vertices is irregular. Examples include the pentagonal pyramid and the elongated square pyramid.

## Polyhedral Dice

Special dice known as polyhedral dice are used in board games, role-playing games, and mathematics games. They are generally applied to games to add an element of chance or randomness.

Polyhedral dice, as opposed to traditional six-sided dice (D6), have more than six faces, enabling a greater range of outcomes.

Some Examples of Polyhedral dice are:

- ****D4:**** This is a tetrahedron-shaped die with four triangular faces.
- ****D6:**** This is the six-sided die most people are familiar with it as we all have played ludo,  snakes, and ladder once in our lifetime.
- ****D8:**** This die has eight triangular faces.
- ****D20:**** The twenty-sided die has twenty equilateral triangular faces.

## Polyhedron Formula

[Euler's formula](https://www.geeksforgeeks.org/maths/eulers-formula/) states that for any convex polyhedron, the following equation holds:

### Euler's formula for Polyhedron

> ****F + V - E = 2****
> 
> Where, 
> 
> - ****F**** is the total number of faces,
> - ****V**** is the total number of vertices, and
> - ****E**** is the total number of edges.

Let's consider an example to verify the above formula.

****Example: Verify the Euler's Formula for Cube.****

****Solution:****

> For a Cube,
> 
> F = 6, E = 12, V = 8  
> Thus, 6 + 8 - 12 = 2
> 
> Therefore, the formula states that the above figure is true and convex polyhedron i.e., Cube.


*[****Polyhedron****](https://www.geeksforgeeks.org/maths/polyhedrons/)

We see various types of figures in our daily life that are shaped like cubes that include, boxes, ice cubes, sugar cubes, etc.

### Cube Examples

Various Cube Examples are shown in the image below:

![Cube-Examples](https://media.geeksforgeeks.org/wp-content/uploads/20240110184049/Cube-Examples.webp)

Cube Examples

### Terms Related to Cube

There are Six Faces, Twelve Edges and Eight Vertices in a Cube.

- ****Cube Faces****

> There are 6 faces in a cube and each faces are have same length and breadth. Hence, a cube has square faces.

- ****Cube Edges****

> There are 12 edges in a cube. Cube Edges mark the boundary of the surfaces of cube.

- ****Cube Vertices****

> There are 8 vertices in a cube. Cube Vertices are coners or the point of intersection of two or more edges.

****Read More:**** [****Faces, Edges and Vertices of Cube****](https://www.geeksforgeeks.org/maths/how-many-faces-edges-and-vertices-does-a-cube-have/)

## Euler's Formula in Cube

Euler's Formula gives the relation between Faces, Edges and Vertices of a Polyhedron. Let's verify the same for a cube. According to Euler's Formula we know that

> ****F + V = E + 2****
> 
> where,
> 
> - F is Number of Faces,
> - V is Number of Vertices
> - E is Number of Edges

In a cube, F = 6, V = 8 and E = 12. Putting this value in above expression we get

> LHS = F + V = 8 + 6 = 14  
> RHS = E + 2 = 12 + 2 = 14
> 
> Hence, F + V = E + 2

Thus, cube satisfies [Euler's Formula](https://www.geeksforgeeks.org/maths/eulers-formula/).

## ****Net of Cube****

A cube is a 3D figure and a figure in 2D that can be folded easily to form the cube is called the net of a cube. Thus, we can say that the two-dimensional form of a cube that can be folded to form a three-dimensional form is called a net of a cube.

There are various ways to unfold a cube, i.e. a cube can have various nets one of nets of cube is discussed in image below,

![Net of Cube](https://media.geeksforgeeks.org/wp-content/uploads/20230526145551/Net-of-Cube.PNG)

## Cube Formula

There are various formulas that are helpful to find various dimensions of cube, that include length of its diagonal, its surface area, its volume, etc. Various cube formulas discussed in article are,

- [Diagonal of Cube](https://www.geeksforgeeks.org/maths/diagonal-of-a-cube-formula/)
- [Surface Area of Cube](https://www.geeksforgeeks.org/maths/surface-area-of-cube/)
- [Volume of Cube](https://www.geeksforgeeks.org/maths/volume-of-cube/)

Now let's learn about these formulas in detail.

### ****Diagonal of Cube****

[Diagonal of a cube](https://www.geeksforgeeks.org/maths/diagonal-of-a-cube-formula/) is the line segment that joins the opposite vertices of the cube. A cube has two types of diagonals, i.e., a face diagonal and a main diagonal.

A face diagonal is a line that joins the opposite vertices of the face of a cube and is equal to the square root of two times the length of the side of a cube. As the cube has six faces, it has a total of 12 face diagonals. The formula to calculate the face diagonal of the cube is,

> ****Length of Face Diagonal of Cube = √2a units****

Where, a is Length of Side of a Cube

While the main diagonal is the [line segment](https://www.geeksforgeeks.org/maths/difference-between-a-line-and-line-segment/) that joins the opposite vertices, passing through the center of the cube, and is equal to the square root of three times the length of the side of a cube. ****A cube has a total of four main diagonals.****

> ****Length of Main Diagonal of Cube = √3a units**** 

Where, a is Length of Side of a Cube

Below image represents main diagonal and face diagonal of cube.

![Diagonal of Cube](https://media.geeksforgeeks.org/wp-content/uploads/20230526145605/Cube2.PNG)

### ****Surface Area of a Cube****

Area of any object is space occupied by all the surfaces of that object. It can be defined as the total surface available for the painting. A cube has six faces and so its surface area is calculated by finding the area of the individual face and finding its sum.

There are two types of surface area associated with a cube that are mentioned below,

- Lateral Surface Area of Cube, also called LSA of Cube.
- Total Surface Area of Cube, also called TSA of Cube.

### ****Lateral Surface Area of Cube****

****Lateral Surface Area of a cube is the sum of the areas of all the faces of a cube, excluding its top and bottom.**** In simple words, the sum of all four side faces of a cube is the lateral surface area of a cube. It is measured in square units such as (units)2, m2, cm2, etc.

****Formula for the lateral surface of a cube is****

> ****Lateral Surface Area of Cube = 4a********2****

where, ****a**** is Length of Side of a Cube

### ****Total Surface Area of Cube****

Total Surface Area of a cube is the space occupied by it in three-dimensional space and is equal to the sum of the areas of all its sides. It is measured in square units such as (units)2, m2, cm2, etc.

****Formula for the total surface of a cube is****

> ****Total Surface Area of Cube = 6a********2****

Where, a is the Length of Side of a Cube

### ****Volume of Cube****

[Volume of a cube](https://www.geeksforgeeks.org/maths/volume-of-cube//) is the amount of space enclosed by the cube. It is usually measured in terms of cubic units. It is measured in cube units such as (units)3, m3, cm3, etc.

Formula for the volume of a cube is

> ****Volume of a Cube = a********3****

Where, a is the Length of Side of a Cube

We can also calculate the volume of the cube if its diagonal is given, by using the formula,

> ****Volume of Cube = (√3d********3********)/9****

where****, d**** is Length of Main Diagonal of Cube.

## ****Properties of Cube****

A cube is a 3D figure with equal dimensions having various properties. Some of the properties of the cube are,

- All the faces of a cube are square-shaped. Hence the length, breadth, and height of a cube are equal.
- The angle between any two faces of a cube is a right angle, i.e., 90°.
- Each face of a cube meets the other four faces.
- Three edges and three faces of a cube meet at a vertex.
- Opposite edges of a cube are parallel to each other.
- Faces or planes of a cube opposite to each other are parallel.

## Interesting Facts about Cube

The various important facts related to cube are mentioned below:

- All the faces of a cube are equal in dimension and is square shaped
- The length, breadth and height of a cube are same
- We can say that cube is a cuboid with equal dimension in all the three directions
- Cube is one of the simplest polyhedrons
- The volume of cube is calculated by side × side × side
- Lateral Surface Area of Cube is calculated by 4 × side2
- Total Surface Area of Cube is Calculated by 6 × side2

****People Also Read:****

> - [Cuboid](https://www.geeksforgeeks.org/aptitude/cuboid/)
> - [Square](https://www.geeksforgeeks.org/maths/square/)
> - [Rectangle](https://www.geeksforgeeks.org/maths/rectangle/)
> - [Cone](https://www.geeksforgeeks.org/)

## ****Cube Examples****

****Example 1: Find the total surface area of a cube if the length of its side is 8 units.****  
****Solution:****

> Given,  
> Length of side of Cube (a) = 8 units
> 
> We know that, ****Total Surface Area of Cube (TSA) = 6a********2****
> 
> TSA = 6 × (8)  
> = 6 ×  
> = 384 square units.
> 
> Hence, the surface area of the cube = 384 square units.

****Example 2: Find the volume of a cube if the length of its side is 5.5 inches.****  
****Solution:****

> Given,  
> Length of side of Cube (a) = 5.5 inches.
> 
> We  know that, ****Volume of Cube (V) = a********3****  
> V = (5.5)  
> = 166.375‬ cubic inches
> 
> Hence, the volume of the cube is 166.375‬ cubic inches.

****Example 3: Find the length of the diagonal of a cube and its lateral surface area if the length of its side is 6 m.****  
****Solution:****

> Given,  
> Length of side of Cube (a) = 6 m
> 
> We know that, ****Length of Diagonal of Cube(l) = √3 a****  
> l = √3 × 6   
> = 6√3 m
> 
> ****Lateral Surface Area of Cube (LSA) = 4a********2****  
> LSA = 4 × (6)  
> = 4 ×  
> = 144 m2  
> Hence, the length of the diagonal is 6√3 m, and its lateral surface area is 144 square meters.

****Example 4: Determine the length of the diagonal of the**** [****cube****](https://www.geeksforgeeks.org/maths/cubes-1-to-20/) ****if the volume of the cube is 91.125 cm********3********.****  
****Solution:****

> Given,  
> Volume of the cube (V) = 91.125 cm3
> 
> Let length of side of a cube be "s"
> 
> We have****, Volume of Cube = s********3****  
> s3 = 91.125  
> s =  ∛(91.125  
> = 4.5 cm
> 
> ****Length of diagonal of a Cube(l) = √3 s****  
> l = √3 × 4.5 = 4.5√3 cm.
> 
> Hence, the length of the diagonal is 4.5√3 cm

| e                                                                          | Figure                                                                                                              | Lateral Surface Area (LSA)              | Total Surface Area (TSA) |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------ |
| [Cube](https://www.geeksforgeeks.org/maths/cube/)                          | ![Surface Area of Cube](https://media.geeksforgeeks.org/wp-content/uploads/20231017205739/Cuboid-(1).png)           | 4a2                                     | 6a2                      |
| [Cuboid](https://www.geeksforgeeks.org/maths/cuboid-shape-and-properties/) | ![Surface Area of Cuboid](https://media.geeksforgeeks.org/wp-content/uploads/20231017205656/Cube-(1).png)           | 2h(l+b)                                 | 2(lb + lh + bh)          |
| [Cylinder](https://www.geeksforgeeks.org/maths/cylinder/)                  | ![Surface Area of Cylinder](https://media.geeksforgeeks.org/wp-content/uploads/20231017205758/Cylinder-(2).png)     | 2πrh                                    | 2πr(r + h)               |
| [Cone](https://www.geeksforgeeks.org/maths/cone/)                          | ![Surface Area of Cone](https://media.geeksforgeeks.org/wp-content/uploads/20231017205824/Cone-(2).png)             | πrl                                     | πr(l + r)                |
| [Sphere](https://www.geeksforgeeks.org/maths/sphere/)                      | ![Surface Area of Sphere](https://media.geeksforgeeks.org/wp-content/uploads/20231017205854/Sphere-(2).png)         | 4πr2                                    | 4πr2                     |
| [Hemisphere](https://www.geeksforgeeks.org/maths/hemisphere/)              | ![Surface Area of Hemisphere](https://media.geeksforgeeks.org/wp-content/uploads/20231017205912/Hemisphere-(2).png) | 2πr2                                    | 3πr2                     |
| [Pyramid](https://www.geeksforgeeks.org/maths/pyramid/)                    | ![Surface Area of Pyramid](https://media.geeksforgeeks.org/wp-content/uploads/20231017205931/Pyramid-(2).png)       | 1/2 × (Base Perimeter) × (Slant Height) | LSA + Area of Base       |
| [Prism](https://www.geeksforgeeks.org/maths/prism/)                        | ![Surface Area of Prism](https://media.geeksforgeeks.org/wp-content/uploads/20231017205952/Prism-(2).png)           | (Base Perimeter) × (Height)             | LSA + 2(Area of Base)    |

## ****Surface Area of Different Shapes in Detail****

Below are the formulas, along with the definitions and figures, for the Lateral Surface Area (LSA) and Total Surface Area (TSA) of various 3D geometrical figures.

### ****Surface Area Formula of Cube****

A cube is a six-faced 3D shape in which all the faces are equal. A cube is a three-dimensional shape with several key characteristics:

![Cube Surface Area](https://media.geeksforgeeks.org/wp-content/uploads/20231010155851/Cube-(2).png)

Formulas for the Surface Area of Cube a are given by:

> - ****Lateral Surface Area (LSA) of Cube =  4a********2****
> - ****Total Surface Area (TSA) of Cube = 6a********2****
> 
> Where ****a**** is the Side of a Cube.

To learn more about the formulas related to a ****Cube****, here are the key ones:

- [Volume of a Cube](https://www.geeksforgeeks.org/maths/volume-of-cube/)
- [Surface Area of Cube](https://www.geeksforgeeks.org/maths/surface-area-of-cube/)

### ****Surface Area Formula of Cuboid****

A Cuboid is a 3D figure in which opposite faces are equal. A cuboid, also known as a rectangular prism, is a 3D geometric shape very similar to a cube, but with some key differences:

![Cuboid Surface Area](https://media.geeksforgeeks.org/wp-content/uploads/20231010155951/Cuboid-(2).png)

Formulas for the Surface Area are given by:

> - ****Lateral Surface Area (LSA) of Cuboid =  2 × (hl + bh)****
> - ****Total Surface Area (TSA) of Cuboid = 2 × (hl + bh + bh)****

Where:

- ****l**** is the length of the Cuboid
- ****b**** is the Breadth of Cuboid
- ****h**** is the Height of the Cuboid

To learn more about the formulas related to a ****Cuboid****, here are the key ones:

> - [Volume of Cuboid](https://www.geeksforgeeks.org/maths/volume-of-cuboid/)
> - [Surface Area of a Cuboid](https://www.geeksforgeeks.org/maths/surface-area-of-cuboid/)

### ****Surface Area Formula of a Sphere****

The sphere is a 3D figure that is similar to a real-life ball. A sphere is a three-dimensional, perfectly round object with several key characteristics:

![Sphere Surface Area](https://media.geeksforgeeks.org/wp-content/uploads/20231010160333/Sphere-(1).png)

The formula for the Surface Area of a Sphere is:

> ****Surface Area of Sphere =  4πr********2****
> 
> Where ****r**** is the Radius of the Sphere.

To learn more about the formulas related to a ****Sphere****, here are the key ones:

> - [Surface Area of a Sphere](https://www.geeksforgeeks.org/maths/surface-area-of-sphere/)
> - [Volume of a Sphere](https://www.geeksforgeeks.org/maths/volume-of-a-sphere/)

### ****Surface Area Formula of a Hemisphere****

The hemisphere is a 3D figure that is half of the Sphere. It is created by slicing it through its center with a flat plane.

![Hemisphere Surface Area](https://media.geeksforgeeks.org/wp-content/uploads/20231010160413/Hemisphere-(1).png)

The formula for the [Area of a Hemisphere](https://www.geeksforgeeks.org/maths/surface-area-of-a-hemisphere/) formula is:

> - ****Curved Surface Area (CSA) of Hemisphere = 2πr********2****
> - ****Total Surface Area (TSA) of Hemisphere = 3πr********2****
> 
> Where ****r**** is the Radius of the Sphere.

To learn more about the formulas related to a ****Hemi-sphere****, here are the key ones:

> - [Area of a Hemisphere](https://www.geeksforgeeks.org/maths/surface-area-of-a-hemisphere/)
> - [Volume of Hemisphere](https://www.geeksforgeeks.org/maths/volume-of-hemisphere/)

### ****Surface Area Formula of a Cylinder****

A cylinder is a 3D figure with two circular bases and a curved surface.

![Cylinder Surface Area](https://media.geeksforgeeks.org/wp-content/uploads/20231010160536/Cylinder-(1).png)

The formula for the Area of a Cylinder is:

> - ****Curved Surface Area (CSA) of Cylinder = 2πrh****
> - ****Total Surface Area (TSA) of Cylinder = 2πr********2**** ****+ 2πrh = 2πr(r+h)****

Where:

- ****r**** is the Radius of the base of the Cylinder
- ****H**** is the Height of Cylinder

To learn more about the formulas related to a ****Cylinder****, here are the key ones:

> - [Area of a Cylinder](https://www.geeksforgeeks.org/maths/volume-of-a-cylinder/)
> - [Volume of a Cylinder](https://www.geeksforgeeks.org/maths/volume-of-a-cylinder/)

### Surface Area Formula of a Cone

A cone is a 3D geometric shape with a circular base and a pointed edge at the top called the apex. A cone has one face and a vertex.

![Cone Surface Area](https://media.geeksforgeeks.org/wp-content/uploads/20231010160738/Cone-(1).png)

The Formula for the Surface Area of the Cone is:

> ****Curved Surface Area (CSA) of Cone = πrl****
> 
> ****Total Surface Area (TSA) of Cone = πr(r + l)****

Where:

- ****r**** is the Radius of the Base of the Cone
- ****l**** is the Slant Height of the Cone

To learn more about the formulas related to a ****Cone****, here are the key ones:

> - [Surface Area of Cone](https://www.geeksforgeeks.org/maths/surface-area-of-cone/)
> - [Volume of a Cone](https://www.geeksforgeeks.org/maths/volume-of-cone/)

### ****Surface Area Formula of Pyramid****

A pyramid is a 3D figure having triangular faces and a triangular base. It is a three-dimensional polyhedron with a polygonal base and triangular sides that meet at a common point called the apex.

![Pyramid Surface Area](https://media.geeksforgeeks.org/wp-content/uploads/20231010160949/Pyramid-(1).png)

The formula for the Surface Area of a Pyramid is:

> - ****Lateral Surface Area (LSA) of Pyramid =  1/2 × (Perimeter of Base) × Height****
> - ****Total Surface Area (TSA) of Pyramid = [1/2 × (Perimeter of Base) × Height] + Area of Base****

To learn more about the formulas related to a ****Pyramid****, here are the key ones:

> - [Surface Area of a Pyramid](https://www.geeksforgeeks.org/maths/surface-area-of-a-pyramid-formula/)
> - [Volume of a Pyramid](https://www.geeksforgeeks.org/maths/volume-of-a-pyramid-formula/)

## Solved Examples of Surface Area Formulas

****Example 1: Find the lateral surface of a Sphere with a radius of 4 cm.****  
****Solution:****

> Given,  
> Radius of Sphere (r) = 4 cm
> 
> Formula of Lateral Surface Area of Sphere = 4πr2  
> LSA = 4 × 3.14 × r × r = 4 × 3.14 × 4 × 4  
> LSA = 200.96 cm2

****Example 2: Find the lateral surface of a Hemi-Sphere with a radius of 6 cm.****  
****Solution:****

> Given,  
> Radius of Hemisphere (r) = 6 cm
> 
> Formula of Lateral Surface Area of Hemi-Sphere  = 2πr2  
> LSA = 2 × 3.14× r × r = 2 × 3.14 × 6 × 6  
> LSA = 226.08 cm2

****Example 3: Find the Total surface of a Cube with a side of 10 m.****  
****Solution:****

> Given,  
> Side of Cube (a) = 10 cm
> 
> Formula of Total Surface Area of Cube = 6a2  
> TSA = 6 × a × a = 6 × 10 × 10  
> TSA = 600 m2

****Related Articles:****

- [Volume Formulas](https://www.geeksforgeeks.org/maths/volume-formulas/)
- [Volume of Cube](https://www.geeksforgeeks.org/maths/volume-of-cube/)
- [Volume of Cylinder](https://www.geeksforgeeks.org/maths/volume-of-a-cylinder/)
- [Volume of Cuboid](https://www.geeksforgeeks.org/maths/volume-of-cuboid/)
- 

- [****Surface Area Formulas****](https://www.geeksforgeeks.org/maths/surface-area-formulas/)
- [****Surface Area of Cone****](https://www.geeksforgeeks.org/maths/surface-area-of-cone/)
- [****Surface Area of Cuboid****](https://www.geeksforgeeks.org/maths/surface-area-of-cuboid/)
- [****Surface Area of a Cube****](https://www.geeksforgeeks.org/maths/surface-area-of-cube/)
- [****Volume of Sphere****](https://www.geeksforgeeks.org/maths/volume-of-a-sphere/)

## How to Find Surface Area of Sphere?

The surface area of a sphere is simply the area occupied by its surface. Let's consider an example to see how to use its formula.

****Example:**** Find the surface area of a sphere of radius 7 cm.

> ****Step 1:**** Note the radius of the given sphere. Here, the radius of the sphere is 47 cm.
> 
> ****Step 2:**** We know that the surface area of a sphere = 4πr2. So, substitute the value of the given radius in the equation = 4 × (3.14) × (7)2 = 616 cm2.
> 
> ****Step 3:**** Hence, the surface area of the sphere is 616 square cm.

## Curved Surface Area (CSA) of Sphere

The sphere has only one curved surface. Therefore, the curved surface area of the sphere is equal to the total surface area of the sphere, which is equal to the surface area of the sphere in general.

Therefore,

> ****CSA of a Sphere = 4πr********2****

## Total Surface Area (****TSA****) of Sphere

As the complete surface of the sphere is curved thus total Surface Area and Curved Surface Area are the same for the Sphere.

> ****TSA of Sphere = CSA of Sphere****

****Must Read:****

> - [Sphere: Definition and Properties](https://www.geeksforgeeks.org/maths/sphere/)
> - [Sphere Formulas](https://www.geeksforgeeks.org/maths/sphere-formulas/)
> - [Radius Formula](https://www.geeksforgeeks.org/maths/radius/)
> - [pi (π) Formula](https://www.geeksforgeeks.org/maths/pi-formulas/)

## Solved Examples on Surface Area of Sphere

Let's solve questions on the Surface Area of Sphere.

****Example 1: Calculate the total surface area of a sphere with a radius of 15 cm. (Take π = 3.14****)

****Solution:****

> Given, the radius of the sphere = 15 cm
> 
> We know that the total surface area of a sphere = 4 π r2 square units
> 
> = 4 × (3.14) × (15)2
> 
> = 2826 cm2
> 
> Hence, the total surface area of the sphere is 2826 cm2.

****Example 2: Calculate the diameter of a sphere whose surface area is 616 square inches. (Take π = 22/7)****

****Solution:****

> Given, the curved surface area of the sphere = 616 sq. in
> 
> We know,
> 
> The total surface area of a sphere = 4 π r2 square units
> 
> ⇒ 4 π r2 = 616
> 
> ⇒ 4 × (22/7) × r2 = 616
> 
> ⇒ r2 = (616 × 7)/(4 × 22) = 49
> 
> ⇒ r = √49 = 7 in
> 
> We know, diameter = 2 × radius = 2 × 7 = 14 inches
> 
> Hence, the diameter of the sphere is 14 inches.

****Example 3: Find the cost required to paint a ball that is in the shape of a sphere with a radius of 10 cm. The painting cost of the ball is ₨ 4 per square cm. (Take π = 3.14)****

****Solution:****

> Given, the radius of the ball = 10 cm
> 
> We know that,
> 
> The surface area of a sphere = 4 π r2 square units
> 
> = 4 × (3.14) × (10)2
> 
> = 1256 square cm
> 
> Hence, the total cost to paint the ball = 4 × 1256 = ₨ 5024/-

****Example 4: Find the surface area of a sphere whose diameter is 21 cm. (Take π = 22/7)****

****Solution:**** 

> Given, the diameter of a sphere is 21 cm
> 
> We know,
> 
> diameter =  2 × radius
> 
> ⇒ 21 = 2 × r ⇒ r = 10.5 cm
> 
> Now, the surface area of a sphere = 4 π r2 square units
> 
> = 4 × (22/7) × (10.5) 
> 
> = 1386 sq. cm
> 
> Hence, the total surface area of the sphere = 1386 sq. cm.

## Practice Problems - Surface Area of Sphere

****Problem 1:**** Find the surface area of a sphere with a radius of 5 cm. Use π=3.14.

****Problem 2:**** A sphere has a diameter of 10 inches. Calculate its surface area.

****Problem 3:**** Determine the surface area of a sphere whose radius is 7 meters.

****Problem 4:**** The radius of a sphere is 15 cm. What is its surface area in square centimeters?

****Problem 5:**** If a sphere with a radius of 8 cm is cut into two hemispheres, what is the surface area of each hemisphere?

****Problem 6:**** A sphere has a surface area of 500 square meters. What is the radius of the sphere?

****Problem 7:**** Calculate the surface area of a sphere with a radius of 12 cm.

****Problem 8:**** The diameter of a sphere is 16 inches. Find its surface area.

****Problem 9:**** A spherical balloon has a radius of 21 cm. Determine its surface area in square centimeters.

****Problem 10:**** A sphere has a radius of 0.5 meters. Find its surface area in square meters.



- [Surface Area of Sphere](https://www.geeksforgeeks.org/maths/surface-area-of-sphere/)
- [Volume of Sphere](https://www.geeksforgeeks.org/maths/volume-of-a-sphere/)

## Radius of Circle Equation

****Equation of circle on the cartesian plane**** with centre (h, k) is given as,

> ****(x − h)********2**** ****+ (y − k)********2**** ****= r********2****

Where (x, y) is the locus of any point on the circumference of the circle and ‘r’ is the radius of the circle.

If the origin (0,0) becomes the centre of the circle then its equation is given as x2 + y2 = r2, then ****Radius of Circle Formula**** is given by :

> ****(Radius) r = √( x********2**** ****+ y********2**** ****)****

## Chord of Circle ****Theorems****

> ****Theorem 1:**** Perpendicular line drawn from the centre of a circle to a chord bisects the chord.

![Chord of Circle Theorem](https://media.geeksforgeeks.org/wp-content/uploads/20230416193453/4-(6).png)

****Given:**** 

Chord AB and line segment OC is perpendicular to AB

****To prove:**** 

AC = BC

****Construction:**** 

Join radius OA and OB

****Proof:****

In ΔOAC and ΔOBC

∠OCA = ∠OCB (OC is perpendicular to AB)

OA = OB     (Radii of the same circle)

OC = OC     (Common Side)

So, by RHS congruence criterion ΔOAC ≅ ΔOBC

Thus, AC = CB (By CPCT)

****Converse of the above theorem is also true.****

> ****Theorem 2:**** Line drawn through the centre of the circle to bisect a chord is perpendicular to the chord.

(For reference, see the Image used above.)

****Given:**** 

C is the midpoint of the chord AB of the circle with the centre of the circle at O

****To prove:**** 

OC is perpendicular to AB

****Construction:**** 

Join radii OA and OB also join OC

****Proof:****

In ∆OAC and ∆OBC

AC = BC (Given)

OA = OB (Radii of the same circle)

OC = OC (Common)

By SSS congruency criterion ∆OAC ≅ ∆OBC 

∠1 = ∠2 (By CPCT)...(1)

∠1 + ∠2 = 180° (Linear pair angles)...(2)

Solving eq(1) and (2)

∠1 = ∠2 = 90° 

Thus, OC is perpendicular to AB.

### ****People Also Read:****

> - [Circle](https://www.geeksforgeeks.org/maths/circles/)
> - [Circumference of Circle](https://www.geeksforgeeks.org/maths/circumference-of-circle/)
> - [Area of Circle](https://www.geeksforgeeks.org/maths/area-of-a-circle/)
> - [Chords of Circle](https://www.geeksforgeeks.org/maths/chords-of-a-circle/)
> - [Segment of Circle](https://www.geeksforgeeks.org/maths/segment-of-a-circle/)
> - [Sector of Circle](https://www.geeksforgeeks.org/maths/sector-of-a-circle/)
> - [Radius of Curvature Formula](https://www.geeksforgeeks.org/maths/radius-of-curvature-formula/)
> - [Properties of Sphere](https://www.geeksforgeeks.org/maths/sphere/)

## Radius of Circle Examples

****Example 1: Calculate the radius of the circle whose diameter is 18 cm.****

****Solution:**** 

> Given,
> 
> - Diameter of the circle = d = 18 cm
> 
> Radius of the circle by using diameter,
> 
> Radius = (diameter ⁄ 2) = 18 ⁄ 2 cm = 9 cm
> 
> Hence, the radius of circle is 9 cm.

****Example 2: Calculate the circle radius when circumference is 14 cm.****

****Solution:****

> Radius of a circle with a circumference of 14 cm can be calculated by using the formula, 
> 
> - Radius = Circumference / 2π
> 
> r = C / 2π
> 
> r = 14 / 2π {value of π = 22/7}
> 
> r = (14 × 7) / (2 × 22)
> 
> r = 98 / 44
> 
> r = 2.22 cm
> 
> Therefore, the radius of the given circle is 2.22 cm

****Example 3: Find the area and the circumference of a circle whose radius is 12 cm. (Take the value of π = 3.14)****

****Solution:**** 

> Given,
> 
> - Radius = 12 cm
> 
> Area of Circle = π r2 = 3.14  × (12)2
> 
> A = 452.6 cm2
> 
> Now Circumference of circle,
> 
> C = 2πr
> 
> C = 2 × 3.14 × 12
> 
> Circumference = 75.36 cm
> 
> Therefore the area of circle is  452.6 cm2 and circumference of circle is 75.36 cm

****Example 4: Find the diameter of a circle, given that area of a circle, is equal to twice its circumference.****

> Given,
> 
> - Area of Circle = 2 × Circumference
> 
> We Know, 
> 
> - Area of the circle = π r2
> - Circumference = 2πr
> 
> Therefore,
> 
> π r2 = 2×2×π×r
> 
> r = 4
> 
> Therefore,
> 
> diameter = 2 × radius
> 
> diameter = 2 × 4 = 8 units

## Practice Questions on Radius of Circle

****Q1. What is the Radius of circle if its Area is 254 cm********2********?****

****Q2. Find the area of circle with circumference 126 units.****

****Q3. Find the diameter of the circle if its radius is 22 cm.****

****Q4. Find the area of the circle with diameter 10 cm.****

----


- [Radius of Circle](https://www.geeksforgeeks.org/maths/radius/)
- [Segment of a Circle](https://www.geeksforgeeks.org/maths/segment-of-a-circle/)
- [Equation of a Circle](https://www.geeksforgeeks.org/maths/segment-of-a-circle/)
- [What is a Circle?](https://www.geeksforgeeks.org/maths/circles/)
- [Area of a Circle](https://www.geeksforgeeks.org/maths/area-of-a-circle/)
- [Sector of a Circle](https://www.geeksforgeeks.org/maths/sector-of-a-circle/)

## Solved Examples on Circumference of Circle

Some examples on Circumference of a circle are,

****Example 1: What is the circumference of a circle with a diameter of 2 cm?****

****Solution:****

> Given, diameter = 2 cm
> 
> By using formula of circumference of a circle,
> 
> C = π × d  
> C = 3.14 × 2  
> C = 6.28 cm

****Example 2: What is the circumference of a circle with a radius of 3 cm?****

****Solution:**** 

> Given, radius = 3 cm
> 
> C = 2 × π × r   
> C = 2 × 3.14 × 3  
> C = 18.84 cm

****Example 3: What is the circumference of a circle with a diameter of 14cm?****

****Solution:**** 

> Given, diameter = 14 cm
> 
> C = π × d  
> C = 3.14 × 14  
> C = 43.96 cm.

****Example 4: What is the circumference of a circle with a radius of 10 cm?****

****Solution:****

> Given, radius = 10 cm
> 
> C = π × 2r   
> C = 3.14 × 2(10)  
> C = 62.8 cm.

## ****Practical Applications of the Circumference of a Circle****

The concept of ****circumference**** extends far beyond academic problems. In the real world, calculating the ****circumference**** helps in:

- ****Wheels and Gears****: To determine how far a wheel or gear moves in one full rotation, we calculate the ****circumference****. This is critical in automotive design, bike manufacturing, and machinery.
- ****Astronomy****: Scientists often calculate the ****circumference of planets**** to better understand their size and rotation. For example, the Earth’s ****circumference**** is approximately 40,075 kilometers at the equator.
- ****Construction and Engineering****: When building structures such as circular columns, pools, or roundabouts, the ****circumference**** is necessary for calculating material usage.

Understanding the ****circumference**** is not only a key math concept but also a practical tool in various industries.

![Example of circumference](https://media.geeksforgeeks.org/wp-content/uploads/20230310162429/Circumference-of-Circle-2.jpg)




It is a ratio of the [****Circumference of the Circle****](https://www.geeksforgeeks.org/maths/circumference-of-circle/) and the Diameter of the Circle. The value of Pi is an irrational number. Thus the exact value of the π is not found yet.

We can also define π as the total number of times the diameter is wrapped around the circumference of any circle. The approximate value of (π) pi is 3.14 or 22/7. The following illustration represents the value of pi and its relation with the circumference and diameter of the circle.

![Value of Pi](https://media.geeksforgeeks.org/wp-content/uploads/20230904110840/Value-of-Pie.webp)

## ****Pi Values in Fraction and Decimal****

We usually express the value of Pi in two ways that are

- Value of Pi in Fraction
- Value of Pi in Decimal

### Approximate Value of Pi

The [value of pi](https://www.geeksforgeeks.org/maths/value-of-pi/) is non-terminating decimal and non-recurring decimal. However, the approximate value of pi (π) is commonly rounded to 3.14. Below is the approx. value of pie in fraction and decimal form.

### ****Also read:**** [Value of PI](https://www.geeksforgeeks.org/maths/value-of-pi/)

### ****Value of Pi (π) in Fractions****

The pi value can be approximated as the fraction of 22/7. It is known that pi is an irrational number which means that the digits after the decimal point are never-ending and are a non-terminating value. Therefore, 22/7 is used for everyday calculations. ‘****π’**** is not equal to the ratio of any two numbers, which makes it an [****irrational number****](https://www.geeksforgeeks.org/maths/irrational-numbers/)****.****

The approximate value of Pi is the value of the Pi in fractions or up to 2 decimals places. As Pi is an irrational number its exact value is not known and so we take the approximate value of Pi in our calculation. The approximate value of Pi in terms of fractions is,

> ****π = 22/7 (Approx)****

### ****Value of Pi (π) in Decimal****

The approximate value of Pi in terms of [decimals](https://www.geeksforgeeks.org/maths/decimals/) is

> ****π = 3.14 (Approx)****

The pi value up to the first 100 decimal places is:

> ****3.14159 26535 89793 23846 26433 83279 50288 41971 69399 37510 58209 74944 59230 78164 06286 20899 86280 34825 34211 70679 . . .****

## Formula of Pi

The formula used to calculate the value of the Pi is

> ****π = C/D****
> 
> Where,
> 
> - ****C**** is the [Circumference of the Circle](https://www.geeksforgeeks.org/maths/circumference-of-circle/)
> - ****D**** is the [****Diameter of a Circle****](https://www.geeksforgeeks.org/maths/diameter-of-a-circle/)

Using this formula we can easily get the value of pi, But as we know pi is an irrational number so its exact value is unknown and we can only find the approximate value of pi using this formula. The value of the Pi found using this formula is 3.14

## How to Calculate Value of Pi?

Pi is an irrational number and it has an infinite number of decimal values that are non-repeating,. There are various methods to calculate the value of pi up to a hundredth of a place The most common method to find the value of pi is taking the ratio of the Circumference of the circle to the diameter of the circle.

> ****π = Circumference of Circle/Diameter of Circle****

Thus, by drawing various circles and then taking the ratio of the Circumference and the diameter of the circle we get the value of the circle. The table added below shows the circumference of the circle, the diameter of the circle and their ratio as well.

|Circumference(C)|Diameter(D)|C/D|
|---|---|---|
|3.1|1|3.1|
|6.24|2|3.12|
|9.378|3|3.126|
|12.5678|4|3.141|
|15.7075|5|3.1415|

As we take higher values of circumference and diameter then we find to get the more accurate value of pi.

## ****Different Values of Pi****

Other then fractions and decimals there are some other values of Pi as well.

### Value of Pi in Degree

The value of Pi in degrees can easily be found using the relation, of the ratio of circumference of the circle and the diameter of the circle. We know that the circumference of the circle is 2πr, and the diameter of the circle is 2r where r is the radius of the circle. Also, incase of the complete circle the angle subtended at the centre of the circle is 360°also we have two half circle in a circle that is divided by a diameter.

Now, then the ratio of the circumference and the diameter gives the value of pi.

2πr/2r = 360°/2

> ****π radians = 180°****

****Also Check,****

- [Is ****pi**** a rational or irrational number?](https://www.geeksforgeeks.org/maths/is-pi-a-rational-or-irrational-number/)
- [Rational Numbers](https://www.geeksforgeeks.org/maths/what-are-rational-numbers/)
- [Whole Numbers](https://www.geeksforgeeks.org/maths/whole-numbers/)
- [Terminating and Non-Terminating Decimals](https://www.geeksforgeeks.org/maths/terminating-and-non-terminating-decimals/)

## ****Solved Examples on Pi Value****

****Example 1: Find the circumference of a circle which has a radius of 12 cm.****

****Solution:****

> Given,
> 
> - Radius of Circle(r) = 12 cm
> 
> Circumference of Circle(C) = 2πr
> 
> Value of Pi = 3.14
> 
> ⇒ C = 2 ⨉ (3.14) ⨉ (12)
> 
> ⇒ C = 75.36 cm

****Example 2: Find the area of a circle which has a radius of 8 cm.****

****Solution:****

> Given,
> 
> - Radius of Circle(r) = 8 cm
> 
> Area of Circle(A) = πr2
> 
> As, Value of Pi = 3.14
> 
> ⇒ A = (3.14) ⨉ (8)2
> 
> ⇒ A = 200.96 cm2

****Example 3: Find the circumference and the area of the circle which has a radius of 9 cm.****

****Solution:****

> Given,
> 
> - Radius of Circle(r) = 9 cm
> 
> Circumference of Circle(C) = 2πr
> 
> Area of Circle(A) = πr2
> 
> As, Value of Pi = 3.14
> 
> ⇒ C = 2 ⨉ (3.14) ⨉ (12)
> 
> ⇒ C = 56.52 cm
> 
> ⇒ A = π ⨉ (9)2
> 
> ⇒ A = 254.34 cm2

## Conclusion

Pi is an important mathematical term which has a constant and non-terminating, irrational value. It has an unending value of 3.14159 and in mathematical solutions, it is rounded off to 3.14 in terms of decimal values, or 22/7 for numerical calculations to make them easier. It is used in multiple scenarios, to calculate the areas and volumes of spheres, hemispheres, cylinders, circles etc. All those geometrical shapes that have a circle involved in their shaping use the concept and formulae regarding pi. Thus, it is important to learn the values and usages of PI since it is one of the governing factors of geometrical mathematics.

## Practice Problems on Value of Pi

****Problem 1: Calculate the circumference of a circle with a radius of 5 units. [Circumference = 2πr.]****

****Problem 2: If the diameter of a circle is 12 inches, what is its circumference? [Use the formula C = πd.]****

****Problem 3: Given the area of a circle is 64 square meters, find the radius. [The formula for the area of a circle is A = πr².]****

****Problem 4: The side of a square is equal to the diameter of a circle. If the circle's area is 144π square units, what is the side length of the square?****

****Problem 5: The Leibniz formula for π alternates signs in an infinite series: π/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - . . . Calculate an approximation of π using the first 10 terms of this series.****

****Problem 6: If the height of a cylinder is 20 centimeters and the value of it's radius across ends is 14 centimeters, calculate it's volume. [The formula for the volume of a cylinder is V = πr²h]****

****Problem 7: If the radius of a sphere is 30 centimeters, then calculate the volume of the sphere. [The formula for the volume of a sphere is V = 4/3πr********3********]****

****Problem 8: If the radius of a sphere is 10 meters, the calculate it's total surface area. [The formula for the total surface area of a sphere is TSA = 4*πr²]****

****Problem 9: Convert 270 degrees into radians. [The formula to convert degrees into radians is degrees×(π/180) = radians]****

****Problem 10: Convert 3π/4 radians into degrees. [The formula to convert radians into degrees is radians × (180/π)= degrees]****


The approximate [value of Pi](https://www.geeksforgeeks.org/maths/value-of-pi/) is 3.14159263539... which is a non-terminating and non-repeating decimal expansion and we know that the non-terminating and non-repeating decimal is an Irrational Number. Hence, Pi is an irrational number.

### What are Irrational Numbers?

[****Irrational numbers****](https://www.geeksforgeeks.org/maths/irrational-numbers/) are a set of numbers that cannot be expressed in fractions or ratios of integers. it can be written in decimals and has endless non-repeating digits after the decimal point.

> ****Irrational numbers cannot be  expressed in the form of p/q, where q ≠0.****

The decimal Expansion of an irrational number is non-terminating and non-repeating. For example 0.1211212111122... is an irrational number that is non-terminating.

### ****People Also Read:****

> - [What are Numbers?](https://www.geeksforgeeks.org/maths/numbers/)
> - [Rational Numbers](https://www.geeksforgeeks.org/maths/rational-numbers/)
> - [Real Numbers](https://www.geeksforgeeks.org/maths/real-numbers/)

## Summary - Is pi a rational or irrational number

Pi (π) is an irrational number. This means it cannot be expressed as a simple fraction and its decimal representation neither ends nor repeats. Pi's digits continue indefinitely without a repeating pattern, which distinguishes it from rational numbers.# Is pi a rational or irrational number?

Last Updated : 23 Jul, 2025

****Is pi a rational or irrational number?**** Pi(π) is a mathematical constant represented by Greek Letter π. Pi is defined as the constant that is equal to the ratio of the circumference of a circle to its diameter. There are two values of pi that we use often, the first is 22/7 and the second is 3.14.

The question of whether ****Is Pi a Rational or Irrational Number**** always arises in the minds of students and creates confusion. Hence, let's get the answer to this question and understand the explanation of it.

Table of Content

- [Pi Definition](https://www.geeksforgeeks.org/maths/is-pi-a-rational-or-irrational-number/#pi-definition)
- [Pi is Rational or Irrational?](https://www.geeksforgeeks.org/maths/is-pi-a-rational-or-irrational-number/#pi-is-rational-or-irrational)
- [Summary - Is pi a rational or irrational number](https://www.geeksforgeeks.org/maths/is-pi-a-rational-or-irrational-number/#summary-is-pi-a-rational-or-irrational-number)

## Pi Definition

> Pi (π) is a mathematical constant representing the ratio of the circumference of a circle to its diameter. It is a transcendental and irrational number, meaning it cannot be expressed as a simple fraction and its decimal representation never ends or repeats.

## Pi is Rational or Irrational?

> ### ****Answer: Pi****(π) ****is an Irrational Number****

### ****Why Pi is an Irrational Number?****

Pi is a mathematical constant that is given as the ratio of the circumference of a circle to the diameter of the circle. Pi is represented by the Greek letter ****π.**** The approximate [value of Pi](https://www.geeksforgeeks.org/maths/value-of-pi/) is 3.14159263539... which is a non-terminating and non-repeating decimal expansion and we know that the non-terminating and non-repeating decimal is an Irrational Number. Hence, Pi is an irrational number.

### What are Irrational Numbers?

[****Irrational numbers****](https://www.geeksforgeeks.org/maths/irrational-numbers/) are a set of numbers that cannot be expressed in fractions or ratios of integers. it can be written in decimals and has endless non-repeating digits after the decimal point.

> ****Irrational numbers cannot be  expressed in the form of p/q, where q ≠0.****

The decimal Expansion of an irrational number is non-terminating and non-repeating. For example 0.1211212111122... is an irrational number that is non-terminating.

### ****People Also Read:****

> - [What are Numbers?](https://www.geeksforgeeks.org/maths/numbers/)
> - [Rational Numbers](https://www.geeksforgeeks.org/maths/rational-numbers/)
> - [Real Numbers](https://www.geeksforgeeks.org/maths/real-numbers/)

## Summary - Is pi a rational or irrational number

Pi (π) is an irrational number. This means it cannot be expressed as a simple fraction and its decimal representation neither ends nor repeats. Pi's digits continue indefinitely without a repeating pattern, which distinguishes it from rational numbers.

> ****Curious about the nature of π? Read:**** [Is π a rational or irrational number?](https://www.geeksforgeeks.org/maths/is-pi-a-rational-or-irrational-number/) 

> ****Note:**** In Set Theory, Rational Number is represented as Q. If Q is a set of Rational Numbers then set of Natural Numbers (N), Whole Numbers (W) and Integers (Z) are all subsets of Rational Number Set.

## ****Representation of Rational Numbers****

Rational Numbers can be represented in following two ways:

1. ****In Fraction Form****
    - In Fraction form we represents in terms of Numerator and Denominator
2. ****In Decimal Form****
    - In Decimals form we represents in terms of Terminating and Repeating Patterns

## Types of Rational Numbers

Rational Numbers can be classified into following Types

1. Standard Form of Rational Numbers
2. Positive Rational Numbers
3. Negative Rational Numbers
4. Terminating Rational Numbers
5. Non Terminating and Repeating Rational Numbers

### Standard Form of Rational Numbers

The standard form of a rational number is defined as having no common factors other than one between the dividend and divisor, and hence the divisor is positive.  
  
For instance, 12/36 is a rational number. However, it can be simplified to 1/3; the divisor and dividend only share one common element. We could say that rational number ⅓ is in a standard form.

### Positive Rational Numbers

Positive Rational Numbers are those in which both numerators and denominators are either positive or negative. In case both numerators and denominators are negative, -1 can be eliminated as common factor which gives us ultimately Positive Rational Number

Example of Positive Rational Numbers are 2/5, -3/-5 etc.

### Negative Rational Numbers

Negative Rational Numbers are those in which either of Numerator or denominator is negative integer.

Example of Negative Rational Number includes -1/2, 3/-4

### ****Terminating Rational Numbers****

Terminating Decimals are the Rational numbers whose decimal representations end or terminate after a certain number of digits.

Rational Number has terminating expansion if the denominator is in the form of 2m × 5n where either of m and n can be zero

#### ****Example of Terminating Decimal****

> - 12/15 = 0.8
> - 3/4=0.75

### ****Non Terminating and Repeating Rational Numbers****

Repeating Decimals are the Rational numbers whose decimal representations have a repeating pattern.

The decimal expansion of non terminating rational number doesn't end. Same digit or group of digits is repeated after fixed interval

#### ****Example of Non Terminating and Repeating Rational Number****

> - 1/3 = 0.3ˉ0.3ˉ
> - 2/7 = 0.\overline{285714}

## Properties of Rational Numbers

- ****Terminating or Repeating Decimals****: When converted to decimal form, rational numbers either end after a few digits (terminating) or start repeating the same digits over and over (repeating).
- ****Additive Identity****: The additive identity for rational numbers is 0, meaning adding 0 to any rational number yields the same number (a+0=a).
- ****Multiplicative Identity****: The multiplicative identity for rational numbers is 1, as multiplying any rational number by 1 leaves it unchanged (a×1=a).
- ****Additive Inverse****: Every rational number a has an additive inverse −a such that a+(−a)=0.
- ****Multiplicative Inverse****: Every non-zero rational number a has a multiplicative inverse 1/a​ such that a×1/a=1, except when a=0 as a does not have a multiplicative inverse.
- ****Closure under Addition, Subtraction, and Multiplication****: The sum, difference, or product of any two rational numbers is also a rational number.
- ****Division Property****: The quotient of two rational numbers is rational provided the divisor is not zero.
- ****Distributive Property****: Rational numbers follow the distributive property: ****a(b+c)=ab+ac.****
- ****Ordering****: Rational numbers can be ordered. For any two rational numbers, you can always say one is larger, smaller, or equal to the other.

## Idetification of Rational Number

To identify a rational number, check if it can be written as a fraction where both the top number (numerator) and the bottom number (denominator) are whole numbers, and the bottom number isn't zero. Rational numbers also have decimal forms that either end after a few digits or repeat a specific pattern.

****For Example: Which of the following numbers are rational numbers?****

****a) -1.75****  
****b) 2/3****  
****c) √5****  
****d) π****

****Solution:****

> ****a)**** -1.75 is a rational number as it  it has a terminating decimal expansion.
> 
> ****b)**** 2/3 is also a rational number as it can be expressed in the form of a ratio of two integers.
> 
> ****c)**** √5 is an irrational number because  it has a decimal expansion with infinitely many digits without any repeatation.
> 
> ****d)**** π is also an irrational number as it has a decimal expansion with infinitely many digits without any repeaatation.
> 
> Thus, only (a) and (b) are the rational numbers out of all the given numbers.

## List of Rational Numbers in Number System

The following are the classification of rational numbers in Number system .

> - All Natural Numbers are Rational Numbers
> - All Whole Numbers are Rational Numbers
> - All Integers are Rational Numbers
> - All Fractions are Rational Numbers
> - All Terminating Decimals are Rational Numbers
> - All Non-Terminating and Repeating Decimals are Rational Numbers.
> - Square roots of all perfect squares are Rational Numbers
> - Cube Roots of all perfect cubes are Rational Numbers.

****Note: All**** [****real numbers****](https://www.geeksforgeeks.org/maths/real-numbers/) ****are not rational numbers but all the rational numbers are real numbers****

## Arithmetic Operations on Rational Numbers

There are four most common operations for Rational Numbers, which includes the following

- [Addition](https://www.geeksforgeeks.org/#:~:text=/)
- [Subtraction](https://www.geeksforgeeks.org/maths/subtraction/#:~:text=)
- [Multiplication](https://www.geeksforgeeks.org/maths/multiplication/#:~:text=)
- [Division](https://www.geeksforgeeks.org/maths/division/)

![Properties and operations of Rational Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230906112942/Properties-of-Rational-Numbers.png)

### Addition of Rational Numbers

Addition of two rational numbers can be done using the following step-by-step method where the addition of 3/4 and 1/6 is explained as an example.

> ****Step 1:**** Find the common denominator (LCD) for both the rational number. i.e.,
> 
> Common denominator for 4 and 6 is ****12****.
> 
> ****Step 2:**** Convert both the rational number to equivalent fractions using the common denominator. i.e.,
> 
> 3/4 = (3 × 3)/(4 × 3) = ****9/12****
> 
> 1/6 = (1 × 2)/(6 × 2) = ****2/12****
> 
> ****Step 3:**** Add numerators of the equivalent fractions obtained in step 2. i.e.,
> 
> 9/12 + 2/12 = (9 + 2)/12 = ****11/12****
> 
> ****Step 4:**** Simplify the resulting fraction if possible. i.e.,
> 
> ****11/12**** is already in its simplest form.
> 
> ****Thus, Addition of 3/4 and 1/6 is 11/12 .****

### Subtraction of Rational Numbers

Subtraction of two Rational Numbers can be done using the following step-by-step method where subtraction of 1/3 and 2/5 is explained.

> ****Step 1:**** Find the common denominator (LCD) for both the rational number. i.e.,
> 
> Common denominator for 3 and 5 is ****15****.
> 
> ****Step 2:**** Convert both the rational numbers to equivalent fractions with the common denominator. i.e.,
> 
> 1/3 = (1 × 5)/(3 × 5) = ****5/15****
> 
> 2/5 = (2 × 3)/(5 × 3) = ****6/15****
> 
> ****Step 3:**** Subtract numerators of equivalent fractions obtained in step 2. i.e.,
> 
> 5/15 - 6/15 = (5 - 6)/15 = -1/15
> 
> ****Step 4:**** Simplify the resulting fraction if possible. i.e.,
> 
> -1/15 is already in its simplest form.
> 
> ****Therefore, 1/3 - 2/5 = -1/15.****

### Multiplication of Rational Numbers

Multiplication of two rational numbers can be achieved by simply multiplying the numerator and denominator of the given Rational Numbers. Step by step method with an example of multiplication of -11/3 and 4/5 is as follows:

> ****Step 1:**** Write both rational number in with multiplication sign(****×)**** in between. i.e. ****-11/3 × 4/5****
> 
> ****Step 2:**** Multiply the numerator and denominator individually. i.e.,****(-11 × 4)/(3 × 5)****
> 
> ****Step 3:**** We get the result of the multiplication. i.e., ****-44/15****

### Division of Rational Numbers

Division of two Rational numbers can be achieved in the following steps(where the division of 3/5 and 4/7 is explained):

> ****Step 1:****  Write both rational number in with division sign in between. i.e., ****3/5 ÷ 4/7****
> 
> ****Step 2:**** Change "÷" with  "×" and take reciprocal of the second rational number. i.e., ****3/5 × 7/4****
> 
> ****Step 3:**** Multiply the numerator and denominator of the resulting fractions. i.e., ****(3 × 7)/(5 × 4)****
> 
> ****Step 4:**** We get the result of the division. i.e., ****21/20****

## Methods to Find Rational Numbers between Two Rational Numbers

Between two rational numbers there exists infinite rational numbers. However, we can find a rational number between two rational numbers using the formula 1/2(a + b) where a and b are rational numbers. Let's say we have to find rational numbers between 2 and 3 then a rational number between 2 and 3 is given as 1/2(2 + 3) = 5⨯1/2 = 5/2

However, other methods also exist to find [rational numbers between two rational numbers](https://www.geeksforgeeks.org/maths/rational-numbers-between-two-rational-numbers-class-8-maths/).

### ****Method 1: To find rational numbers between two rational numbers with like denominators.****

> In this, we need to multiply the numerator and denominator of rational numbers with a larger number to create a gap between numerators.
> 
> Once the gap is created write the in-between rational numbers just increasing the numerator by 1 and keeping the denominators same.

****Example: Find 10 rational numbers between 4/5 and 6/5.****

****Solution:****

> In this case, we see that we can only find only one rational number between 4/5 and 6/5 which is 5/5. But here we need to find 10 rational numbers.
> 
> Hence, we would multiply the numerator and denominator in both the rational number by 10. Hence we have to find 10 rational numbers between (4⨯10)/(5⨯10) and (6⨯10)/(5⨯10) i.e. 40/50 and 60/50.
> 
> Hence, ten rational numbers between 40/60 and 50/60 are 41/50, 42/50, 43/50, 44/50, 45/50, 46/50, 47/50, 48/50, 49/50, 50/50.
> 
> If we need more we would multiply by a larger number. For simplicity, you can multiply by 10, 100, etc.

### ****Method 2: To find a rational number between two rational numbers with unlike denominators****

> In this, we first convert the unlike denominators to like decimals then follow the same method as followed in the case of like denominators

****Example: Find 5 rational numbers between 4/3 and 6/5****

****Solution:****

> Here we will first make the denominators like, by taking the LCM of denominators 3 and 5. Hence, the LCM of 3 and 5 is 15. Therefore our new equivalent rational numbers will be (4⨯5)/(3⨯5) and (6⨯3)/(5⨯3) i.e. 20/15 and 18/15.
> 
> Still, we see that gap is of two only between 18 and 20. Hence, we will multiply with a larger number say 5.
> 
> Hence, we have to find a rational number between 20⨯5/15/⨯5 and 18⨯5/15⨯5 i.e. 100/75 and 90/75. Hence, 5 rational numbers between 90/75 and 100/75 are 91/75, 92/75, 93/75, 94/75 and 95/75.

### ****Method 3: To find 'n' rational numbers between two rational numbers x and y with unlike denominators such that x < y****

> In this case, first calculate, d = (y - x)/(n + 1) then find the rational numbers between two rational numbers as (x + d), (x + 2d), (x + 3d),.....,(x + nd)

****Example: Find five rational numbers between 1/3 and 2/5.****

****Solution:****

> x = 1/3, y = 2/5, n = 5
> 
> d = (y - x)/(n + 1) = (2/5 - 1/3)/(5 + 1) = 1/15/6 = 1/90
> 
> Five rational numbers between 1/3 and 2/5 are given as
> 
> (x + d), (x + 2d), (x + 3d), (x + 4d) and (x + 5d)
> 
> (1/3 + 1/90), (1/3 + 2/90), (1/3 + 3/90), (1/3 + 4/90) and (1/3 + 5/90)
> 
> (31/90), (32/90), (33/90), (34/90) and (35/90)

## Representing Rational Numbers on Number Line

Rational Numbers are Real Numbers. Hence they can be represented on real line. There are following steps that need to be followed to represent rational numbers on real line.

> - ****Step 1:**** First find if the number is positive or negative if positive then it will be plotted on the RHS of zero and if positive it will be on the LHS of zero.
> - ****Step 2:**** Identify if the given rational number is proper or improper. If proper then it will lie between 0 and 1 in case of positive and 0 and -1 in case of negative rational number.
> - ****Step 3:**** If improper then convert it into a mixed fraction. In this case, the rational number will lie just beyond the whole number part.
> - ****Step 4:**** Now after steps 1, 2, and 3 we have to plot only the proper fraction part. To plot this cut the area between the two successive desired numbers by drawing lines n-1 times where n is the denominator of the proper fraction part.
> - ****Step 5:**** Now count the lines equal to the value of the numerator. This will represent the desired rational number on a real line.

Let's see some examples:

****Example 1: Represent 2/5 on Real Line****

****Solution:****

![Rational Numbers - Represent 2/5 on Real Line](https://media.geeksforgeeks.org/wp-content/uploads/20230825183331/Screenshot-from-2023-08-01-18-13-51.png)

****Example 2: Represent -2/5 on Real Line****

****Solution:****

![Rational Numbers - Represent -2/5 on Real Line](https://media.geeksforgeeks.org/wp-content/uploads/20230825183558/Screenshot-from-2023-08-01-18-13-56.png)

****Example 3: Represent 4/3 on Real Line****

****Solution:****

4/3 = 1(1/3)

![Rational Numbers - Represent 4/3 on Real Line](https://media.geeksforgeeks.org/wp-content/uploads/20230825183837/Screenshot-from-2023-08-01-18-14-01-(1).png)

****Example 4: Represent -4/3 on Real Line****

****Solution:****

-4/3 = -{1(1/3)}

![Rational Numbers - Represent 4/3 on Real Line](https://media.geeksforgeeks.org/wp-content/uploads/20230825184034/Screenshot-from-2023-08-01-18-14-06.png)

## Difference Between Rational and Irrational Numbers

[Irrational Numbers](https://www.geeksforgeeks.org/maths/irrational-numbers/) are those which can't be represented in the form of p/q where q ≠ 0. The decimal expansion of irrational numbers is non-terminating and non-repeating. [Diffrence between rational and irrational numbers](https://www.geeksforgeeks.org/maths/what-is-the-difference-between-rational-and-irrational-numbers/) in the table given below:

|Rational Numbers|Irrational Numbers|
|---|---|
|It can be represented in the form of p/q where q ≠ 0|It can't be represented in the form of p/q where q ≠ 0|
|Its Decimal Expansion is either terminating or non-terminating and repeating|Its Decimal Expansion is non-terminating and non-repeating|
|A set of rational numbers contains all types of numbers such as natural numbers, whole numbers, and integers.|Irrational Numbers doesn't contain all types of numbers in itself|
|Examples include 2/3, -5/6, 0.25, 0.333, 22/7, etc.|Examples include √2,√3, 1.010010001, π, etc.|

|****Articles related to Rational Numbers****|   |
|---|---|
|[Natural Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/)|[Whole Numbers](https://www.geeksforgeeks.org/maths/whole-numbers/)|
|[Prime Numbers](https://www.geeksforgeeks.org/maths/prime-number/)|[Imaginary Numbers](https://www.geeksforgeeks.org/maths/imaginary-numbers/)|

## Rational Numbers Examples

****Example 1: Check which of the following is irrational or rational: 1/2, 13, -4, √3, and π.****

****Solution:****

> Rational numbers are numbers that can be expressed in the form of p/q, where q is not equal to 0.
> 
> 1/2, 13, and -4 are rational numbers as they can be expressed as p/q.
> 
> √3, and π are irrational numbers as they can not be expressed as p/q.

****Example 2: Check if a mixed fraction, 3(5/6) is a rational number or an irrational number.****

****Solution:****

> Simplest form of 3(5/6) is 23/6
> 
> Numerator = 23, which is an integer
> 
> Denominator = 6, is an integer and not equal to zero.
> 
> So, 23/6 is a rational number.

****Example 3: Determine whether the given numbers are rational or irrational.****

****(a) 1.33  (b) 0.1  (c) 0  (d) √5****

****Solution:****

> ****a) 1.33**** is a rational number as it can be represented as 133/100.
> 
> ****b) 0.1**** is a rational number as it can be represented as 1/10.
> 
> ****c) 0**** is a rational number as it can be represented as 0/1.
> 
> ****d) √5**** is an irrational number as it can not be represented as p/q.

****Example 4: Simplify (2/3) × (6/8) ÷ (5/3).****

****Solution:****

> (2/3) × (6/8) ÷ (5/3) = (2/3) x (6/8) × (3/5)
> 
> = (2 × 6 × 3)/(3 × 8 × 5) 
> 
> = 36/120 = 3/10

****Example 5: Arrange following rational numbers in ascending order: 1/3, -1/2, 2/5, and -3/4.****

****Solution:****

> Common denominator for 3, 2, 5, and 4 is 60. Thus
> 
> 1/3 = 20/60
> 
> -1/2 = -30/60
> 
> 2/5 = 24/60
> 
> -3/4 = -45/60
> 
> With common denominator, rational number with greatest numerator is greatest.
> 
> ⇒ -30/60 < -45/60 < 20/60 < 24/60
> 
> Thus, ascending order of given rational numbers is: -1/2 < -3/4 < 1/3 < 2/5

## Rational Numbers Worksheet

Try out the following questions on rational numbers

****Q1. Find two rational number between 2/3 and 3/4****

****Q2. Find the sum of -3/5 and 6/7****

****Q3. Find the first five equivalent rational numbers of -7/8****

****Q4. Represent 4/3 on Real Line****

****Q5. Find the Product of -19/3 and 2/57****

### Rational Numbers Worksheet PDF

You can download this worksheet from below with answers:

|[Download Rational Numbers Worksheet](https://media.geeksforgeeks.org/wp-content/uploads/20241007041610536349/Rational-Numbers-Worksheet_compressed.pdf)|
|---|

## Summary

Rational numbers are essentially any numbers that can be expressed as a fraction or ratio where both the numerator (the top number) and the denominator (the bottom number) are integers, and the denominator is not zero. This category includes a wide array of numbers such as [natural numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/), [whole numbers](https://www.geeksforgeeks.org/maths/whole-numbers/), [integers](https://www.geeksforgeeks.org/maths/integers/), [fractions](https://www.geeksforgeeks.org/maths/fractions/), and even some decimal numbers if they terminate (come to an end) or repeat a pattern.

They are fundamental in [mathematics](https://www.geeksforgeeks.org/maths/maths/) because they allow us to perform arithmetic operations like addition, subtraction, multiplication, and division, and they can represent a wide range of quantities in real life.


----------------


> - [Rational Numbers](https://www.geeksforgeeks.org/maths/rational-numbers/): Q = { {ab∣a,b∈Z,b≠0}{ba​∣a,b∈Z,b=0} }  
>     (All numbers that can be expressed as a fraction abba​ ​, where a and b are integers and b ≠ 0.)

Let's learn about the ****definition, symbols, properties, and examples of whole numbers in detail, along with some numerical examples and worksheets.****

It can be said that the whole number is a set of [numbers](https://www.geeksforgeeks.org/maths/numbers/) without fractions, decimals, and negative numbers.

### ****Whole Number Symbol****

The symbol to represent whole numbers is the alphabet ****‘W’**** in capital letters.

> The ****whole numbers list**** includes 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, to infinity.
> 
> ****W = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,…}****

### Set of Whole Numbers

The set of whole numbers includes the set of natural numbers along with the number 0. The set of whole numbers in mathematics is given as {0, 1, 2, 3, ...}

> - All whole numbers come under real numbers.
> - All natural numbers are whole numbers but not vice-versa.
> - All positive integers, including 0, are whole numbers.
> - All counting numbers are whole numbers.
> - Every whole number is a rational number.

## Whole Numbers on the Number Line

Whole numbers can be easily observed on the number line. They are represented as a collection of all the positive integers, along with 0.

The visual representation of whole numbers on the number line is given below:

![Whole Numbers on Number Line](https://media.geeksforgeeks.org/wp-content/uploads/20230410165350/1-(12).png)

## Properties of Whole Numbers

A Whole Number has the following key properties:

- Closure Property
- Commutative Property
- Associative Property
- Distributive Property

|****Property****|****Description****(where W is a whole number)|
|---|---|
|****Closure Property****|x + y = W OR x × y = W|
|****Commutative Property of Addition****|x + y = y + x|
|****Commutative Property of Multiplication****|x × y = y × x|
|****Additive Identity****|x + 0 = x|
|****Multiplicative Identity****|x × 1 = x|
|****Associative Property****|x + (y + z) = (x + y) + z OR x × (y × z) = (x × y) × z|
|****Distributive Property****|x × (y + z) = (x × y) + (x × z)|
|****Multiplication by Zero****|a × 0 = 0|
|****Division by Zero****|a/0 is undefined|

Let's discuss them in detail.

### Closure Property

The sum and the product of two whole numbers will always be a whole number. 

> ****x + y = W****  
> ****x × y = W****

****For example:**** Prove the closure property for 2 and 5. 

> 2 is a whole number, and 5 is a whole number. To prove the closure property, add and multiply 2 and 5.  
> 2 + 5 = 7 (Whole number).  
> 2 × 5 = 10 (Whole number).

### Commutative Property of Addition

In the commutative property of addition, the sum of any two whole numbers is the same. i.e., the order of addition doesn't matter. i.e., 

> ****x + y = y + x****

****For Example:**** Prove the commutative property of addition for 5 and 8.

> According to the commutative property of addition:  
> x + y = y + x  
> 5 + 8 = 13  
> 8 + 5 = 13
> 
> Therefore, 5 + 8 = 8 + 5

### Commutative Property of Multiplication

The multiplication of any two whole numbers is the same. Any number can be multiplied in any order. i.e.,    

> ****x × y = y × x****

****For example:**** Prove the commutative property of multiplication for 9 and 0.

> According to the commutative property of multiplication:  
> x + y = y + x  
> 9 × 0 = 0  
> 0 × 9 = 0
> 
> Therefore, 9 × 0 = 0 × 9

### Additive Identity

In the additive property, when we add the value with zero, then the value of the integer remains unchanged. i.e., 

> ****x + 0 = x****

****For example:**** Let's prove the numbers property for 7.

> According to additive property  
> x + 0 = x  
> 7 + 0 = 7
> 
> Hence, proved.

### Multiplicative Identity

When we multiply a number by 1, then the value of the integer remains unchanged. i.e., 

> ****x × 1 = x****

****For example:**** Prove multiplicative property for 13.

> According to multiplicative property:  
> x × 1 = x  
> 13 × 1 = 13
> 
> Hence, proved.

### Associative Property

When adding and multiplying the [numbers](https://www.geeksforgeeks.org/maths/numbers/) and grouped together in any order, the value of the result remains the same. i.e.,

>  ****x + (y + z) = (x + y) + z****   
> ****and****   
> ****x × (y × z) = (x × y) ×  z****

****For example:**** Prove the associative property of multiplication for the whole numbers 10, 2, and 5.

> According to the associative property of multiplication:  
> x × (y × z) = (x × y) ×  z  
> 10 × (2 × 5) = (10 × 2) × 5  
> 10 × 10 = 20 × 5  
> 100 = 100
> 
> Hence, Proved.

### Distributive Property

When multiplying the numbers and distributing them in any order, the value of the result remains the same. i.e., 

> ****x × (y + z) = (x × y) + (x × z)****

****For example:**** Prove the distributive property for 3, 6, and 8.

> According to the distributive property:  
> x × (y + z) = (x × y) + (x × z)  
> 3 × (6 + 8) = (3 × 6) + (3 × 8)  
> 3 × (14) = 18 + 24  
> 42 = 42
> 
> Hence, Proved.

### Multiplication by Zero

Multiplication by the zero is a special multiplication as multiplying any [number](https://www.geeksforgeeks.org/maths/numbers/) by zero yields the result zero. i.e.

> ****a × 0 = 0****

****For example:**** Find 238 × 0.

> = 238 × 0  
> we know that multiplying any number yield the result zero.  
> = 0

### Division by Zero

Division is the inverse operation of multiplication. But division by zero is undefined, we can not divide any number by zero, i.e. 

> ****a/0 is undefined****

****Read More :****

- [Properties of Whole Numbers](https://www.geeksforgeeks.org/maths/properties-of-whole-numbers/)
- [Distributive Property](https://www.geeksforgeeks.org/maths/distributive-property/)

## Whole Numbers and Natural Numbers

A natural number is any whole number that is not ****zero.**** Furthermore, all natural numbers are whole numbers. Therefore, the set of natural numbers is a part of the set of whole numbers.

### Whole Numbers vs Natural Numbers

Let's discuss the difference between natural numbers and whole numbers.

|****Whole Numbers vs. Natural Numbers****|   |
|---|---|
|Natural Numbers|Whole Numbers|
|---|---|
|****Smallest natural number is 1.****|****Smallest whole number is 0.****|
|****Set of natural numbers (N) is {1, 2, 3, ...}.****|****Set of whole numbers (W) is {0, 1, 2, 3, ...}****|
|****Every natural number is a whole number.****|****Every whole number is not a natural number.****|

### Whole Numbers vs Integers

Two important sets of numbers you’ll often encounter are ****whole numbers**** and ****integers****. Differences Between Whole Numbers and Integers are given below:

|Feature|Whole Numbers|Integers|
|---|---|---|
|Includes zero|Yes|Yes|
|Positive numbers|Yes|Yes|
|Negative numbers|No|Yes|
|Decimals/Fractions|No|No|
|Set notation|{0, 1, 2, 3, .....}|{…,−3,−2,−1, 0 , 1 , 2 , 3 , .....}|

## Whole Numbers Operations

Whole numbers are the foundation of arithmetic. Understanding how they work with basic operations—****addition, subtraction, multiplication, and division****—is essential for mastering math skills used in school and everyday life.

### Addition – Combining Quantities

Addition is the process of putting two or more numbers together to make a larger total.Example: 7 + 5 = 12  
Properties of Addition:

> - ****Commutative:**** a + b = b + a
> - ****Associative:**** (a + b) + c = a + (b + c)
> - ****Identity Element:**** a + 0 = a

### Subtraction – Finding the Difference

****Subtraction**** is used to find how much one number is greater than another, or how much is left when something is taken away.  
Example: 9 - 4 = 5

### Multiplication – Repeated Addition

****Multiplication**** is a quick way to add the same number multiple times.  
Example: 6 × 3 = 18  
Properties of Multiplication:

> - ****Commutative:**** a × b = b × a
> - ****Associative:**** (a × b) × c = a × (b × c)
> - ****Identity Element:**** a × 1 = a
> - ****Zero Property:**** a × 0 = 0

### Division – Splitting into Equal Parts

****Division**** is the process of splitting a number into equal groups or parts.

****Example:**** 12 / 3 = 4

****Read More:****

> - [Whole Numbers vs Natural Numbers](https://www.geeksforgeeks.org/maths/difference-between-natural-and-whole-numbers/)
> - [Natural Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/)

## Solved Question on Whole Numbers

****Question 1:**** Are the numbers 100, 399, and 457 whole numbers?

****Solution:****

> Yes, the numbers 100, 399, 457 are the whole numbers.

****Question 2:**** Solve the equation 15 × (10 + 5) using the distributive property.

****Solution:****

> We know that distributive property are:
> 
>  x × (y + z) = x × y + x × z
> 
> So, 15 × 10 + 15 × 5 = 150 + 75 
> 
> = 225.

****Question 3:**** Prove the associative property of multiplication for the whole numbers 1, 0, and 93.

****Solution:****

> According to the associative property of multiplication:
> 
> x × (y × z) = (x × y) ×  z
> 
> 1 × (0 × 93) = (1 × 0) × 93
> 
> 1 × 0 = 0 × 93
> 
> 0 = 0
> 
> Hence, Proved.

****Question 4:**** Write down the number that does not belong to the whole numbers:

4, 0, -99, 11.2, 45, 87.7, 53/4, 32.

****Solution:****

> Out of the numbers mentioned above, it can easily be observed that 4, 0, 45, and 32 belong to whole numbers. Therefore, the numbers that do not belong to whole numbers are -99, 11.2, 87.7, and 53/4.

****Question 5:**** Write 3 whole numbers occurring just before 10001.

****Solution:****

> If the sequence of whole numbers are noticed, it can be observed that the whole numbers have a difference of 1 between any 2 numbers. Therefore, the whole numbers before 10001 will be: 10000, 9999, 9998.

### ****Related Articles:****

- [Smallest Whole Number](https://www.geeksforgeeks.org/maths/which-is-the-smallest-whole-number/)
- [Real Numbers](https://www.geeksforgeeks.org/maths/real-numbers/)
- [Rational Numbers](https://www.geeksforgeeks.org/maths/rational-numbers/)
- [Irrational Numbers](https://www.geeksforgeeks.org/maths/irrational-numbers/)
- [Complex Numbers](https://www.geeksforgeeks.org/maths/complex-numbers/)

## Whole Numbers Worksheet

![Screenshot-2024-10-07-035226](https://media.geeksforgeeks.org/wp-content/uploads/20241007035323207125/Screenshot-2024-10-07-035226.png)

You can download this worksheet from below with answers:

|[Download Whole Numbers Worksheet](https://media.geeksforgeeks.org/wp-content/uploads/20241007035117155094/Whole-Numbers-Worksheet_compressed.pdf)|
|---|

## Conclusion

The set of natural numbers that includes zero is known as ****whole numbers: 0, 1, 2, 3, 4,**** and so on. In terms of whole numbers, they are ****non-negative integers,**** which means that they begin at zero and go indefinitely in a positive direction without containing fractions or decimals. In many mathematical operations****, including counting, addition, subtraction, multiplication, and division, whole numbers are necessary****. Understanding the characteristics and functions of whole numbers is essential in the teaching of mathematics and ****establishes the foundation for additional mathematical exploration.****

After the discovery of 0 Whole Numbers became the natural continuation of Natural Numbers. As [Whole Numbers](https://www.geeksforgeeks.org/maths/whole-numbers/) are defined as the collection of Natural Numbers including 0 i.e., 0, 1, 2, 3, 4, . . . and goes on forever. Whole Numbers are represented by the letter W. The image added below shows Whole Numbers.

![Whole Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230505075939/Numbers-1.png)

### Integers

When the use of negative numbers was popularized, they were very useful for many real-life use cases, such as debt-oriented calculations. [Integers](https://www.geeksforgeeks.org/maths/integers/) came into existence, as these are collections of whole numbers as well as the negative of each natural number, i.e., . . . -4, -3, -2, -1, 0, 1, 2, 3, 4, . . .,  and these go forever on both sides. Integers are represented by Z.

> ****Also read:**** [****Interesting Facts about Integers****](https://www.geeksforgeeks.org/maths/interesting-fact-about-integers/)

### Rational Numbers

There was a problem in ancient Egypt with how to represent half or one-third of something in the records, so they came up with the solution known as fractions, and these fractions further evolved into [Rational Numbers](https://www.geeksforgeeks.org/maths/rational-numbers/) as we know them today. For a definition, Rational Numbers are those numbers that can be represented in the p/q form, where p and q are both integers and q can never be 0. For example, 1/2, 3/5, 17/41, 13/7, etc. (As we can't list all rational numbers as a list of natural numbers or integers).

The image added below shows the rational and irrational number

![Rational and Irrational Number](https://media.geeksforgeeks.org/wp-content/uploads/20230512125123/Numbers-2--1.webp)

### Irrational Numbers

[Irrational Numbers](https://www.geeksforgeeks.org/maths/irrational-numbers/) came into existence due to geometry, as Pythagoras discovered a very elegant solution for a right-angled triangle known as the [Pythagoras Theorem](https://www.geeksforgeeks.org/maths/pythagoras-theorem/). If there is a right-angled triangle with its base and height both being 1 unit, then using Pythagoras' theorem, its hypotenuse comes to be √2, which back then wasn't known as anything.

> Also there was a dark story about it that goes like one of the Pythagoras's disciple named Hippasus of Metapontum proved the existence of irrational numbers representing √2 as [fraction](https://www.geeksforgeeks.org/maths/fractions/) and proofing that it is a contradiction but Pythagoras believed in the absoluteness of numbers and couldn't accept the existence of irrational number but he also didn't able to disproof logically that irrational numbers doesn't exist.
> 
> So, he sentenced Hippasus' death by drowning to impede the spread of such things which were against the philosophies of Pythagoras.

Irrational numbers are defined as such numbers that can't be represented as the ratio of two integers and are represented by P. Irrational Numbers are non-terminating and non-repeating in nature i.e. they don't have decimal value limited to finite places and also the preparation of digits in decimal expansion is not periodic. Some examples of Irrational Numbers include √2, √3, √11, 2√2,  π(pi), etc. (As we can't list all rational numbers as a list of natural numbers or integers).

### Real Numbers

The collection of rational and irrational numbers is known as [Real Numbers](https://www.geeksforgeeks.org/maths/real-numbers/) but the name comes from the fact that they can be represented on the real number line. All the examples of rational and irrational numbers are examples of Real Numbers as well. All the numbers except imaginary numbers are included under Real Numbers. Real Numbers are represented by R.

### Imaginary Numbers

For a long period of time, people thought that the number system was incomplete and needed some new sort of numbers to complete it, as there was no solution to the equation x2+a=0(where a>0) in real numbers, but we now know by the fundamental theorem of algebra that every polynomial of degree n needs to have n roots. So there must be a new sort of number needed to find the solution to the above equation. 

The solution of the equation x2 + a = 0 is simply x = ±√-a, which in ancient times was not accepted as the solution because they didn't know any such number whose square was a negative number, but eventually, some mathematicians started using such a number and saw that this made sense for a lot of other calculations as well. Some things that mathematicians saw as impossible before the use of the square root of negative numbers now seem graspable. One of the first mathematicians to use this notion was Rafael Bombelli, an Italian mathematician. Eventually, this concept of using the square root of negative numbers is becoming a useful tool for many fields of mathematics as well as physics.

A new symbol, "i(iota)" was used by Euler first for -1 so he could easily represent an [imaginary number](https://www.geeksforgeeks.org/maths/imaginary-numbers/) without writing √-1 repetitively, and it spread across the world and became second nature to use "i" for √-1. Numbers that give a negative value when squared are generally called Imaginary Numbers. Some examples of these numbers are √-21(which can be written as √-1×√21 or i√21), i, 2i, etc.  

### Complex Numbers

[Complex numbers](https://www.geeksforgeeks.org/maths/complex-numbers/) are the result of the endeavor of hundreds of mathematicians to complete the number system and are defined in the form of a+ib, where a and b are real numbers and "i" is the iota, which represents √-1. Complex numbers are represented by C and are the most useful in the different fields of modern physics, such as quantum mechanics and electromagnetic waves. Some examples of Complex Numbers are 1+i, √2-3i, 2-i√5, etc. The image added below represents the general structure of complex numbers.

![Representation of Complex Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230505080057/Numbers-12.png)

The set of numbers is discussed in the image added below which explains that all the numbers known to humans are the subset of complex numbers.

![Numbers represented as set](https://media.geeksforgeeks.org/wp-content/uploads/20230802185652/Numbers-(2).png)

## Numbers in Words

Representation of numerical values in the form of words is referred to as Numbers in Word. In this representation each digit of a number is replaced by a word, for example, the number 231 is represented as "two hundred thirty-one".

In each number system, we have different rules for writing numbers in words. For example, in the Indian system, we separate numerals after every two digits of a number, and the names of successive higher numbers are as follows: one, ten, hundred, thousand, ten thousand, lac, ten lac, crore, ten crores, etc.

But in the international system, digits are separated according to their place value, and the names of successive higher numbers are as follows: one, ten, hundred, thousand, ten thousand, hundred thousand, million, ten million, hundred million, billion, etc.

List of first forty numbers as words as follows:

|   |   |   |   |
|---|---|---|---|
|****1****|One|****21****|Twenty-One|
|****2****|Two|****22****|Twenty-Two|
|****3****|Three|****23****|Twenty-Three|
|****4****|Four|****24****|Twenty-Four|
|****5****|Five|****25****|Twenty-Five|
|****6****|Six|****26****|Twenty-Six|
|****7****|Seven|****27****|Twenty-Seven|
|****8****|Eight|****28****|Twenty-Eight|
|****9****|Nine|****29****|Twenty-Nine|
|****10****|Ten|****30****|Thirty|
|****11****|Eleven|****31****|Thirty-One|
|****12****|Twelve|****32****|Thirty-Two|
|****13****|Thirteen|****33****|Thirty-Three|
|****14****|Fourteen|****34****|Thirty-Four|
|****15****|Fifteen|****35****|Thirty-Five|
|****16****|Sixteen|****36****|Thirty-Six|
|****17****|Seventeen|****37****|Thirty-Seven|
|****18****|Eighteen|****38****|Thirty-Eight|
|****19****|Nineteen|****39****|Thirty-Nine|
|****20****|Twenty|****40****|Forty|

The image added below tells us how we write a number in words

![Numbers in Words](https://media.geeksforgeeks.org/wp-content/uploads/20230505080030/Numbers-7.png)

****Read More about**** [****Number Names****](https://www.geeksforgeeks.org/maths/number-names-1-to-100/)****.****

## Operations on Numbers

Operations on Numbers are the most fundamental building block of mathematics and are used to manipulate numerical values. These operations are as follows:

### Addition

The [addition](https://www.geeksforgeeks.org/maths/addition/) is the most basic operation which combines two or more numbers to get a sum. For example, 2 + 3 = 5, 3+1/2=7/2, 2.14 + 3.73 = 5.87, etc. The image added below shows the addition between 21 + 3 = 24.

![Addition](https://media.geeksforgeeks.org/wp-content/uploads/20230512125122/Numbers-2-2.webp)

### Subtraction

[Subtraction](https://www.geeksforgeeks.org/maths/subtraction/) is used for finding the difference between two numbers. For example, 6 - 4 = 2, 3-1.5 = 1.5, 3/2-1/3 = 7/6, etc. The image added below shows the addition between 21 - 3 = 18.

![Subtraction](https://media.geeksforgeeks.org/wp-content/uploads/20230512125122/Numbers-2-3.webp)

### Multiplication

[Multiplication](https://www.geeksforgeeks.org/maths/multiplication/) is used when we need to add a number a certain number of times, so instead of repeatedly adding it, we just multiply it by the number of times it needs to be added. For example, 3 x 4 = 12 means 3 + 3 + 3 + 3 = 12. The image added below shows the multiplication between 21 × 3 = 63.

![Multiplication](https://media.geeksforgeeks.org/wp-content/uploads/20230512125121/Numbers-2-4.webp)

### Division

In [Division](https://www.geeksforgeeks.org/maths/division/), we can divide a number into equal parts. For example, 12 ÷ 3 = 4 means 12 can be shared into 3 equal parts, each part is 4. The image added below shows the division between 21 ÷ 7 =3. 

![Division](https://media.geeksforgeeks.org/wp-content/uploads/20230512125120/Numbers-2-5.webp)

## Types of Numbers

Apart from the above types, there are some types of numbers that are based on the properties of the numbers. Some of these numbers are,

- Cardinal and Ordinal Numbers
- Even and Odd Numbers
- Prime and Composite Numbers
- Co Prime Numbers
- Perfect Numbers
- Algebraic and Transcendental Numbers

### Cardinal and Ordinal Numbers

Cardinal Numbers are the same as natural numbers as they were defined as sequentially going numbers that start from 1 and go on forever i.e. 1, 2, 3, 4, . . . and so on.

An ordinal Number is a number that shows the exact position or the order of the object in the sequence. For example first, second, third, and so on. The Cardinal and [Ordinal Numbers](https://www.geeksforgeeks.org/maths/ordinal-numbers/) from 1 to 10 are discussed in the  image below,

![Cardinal and Ordinal Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230505080024/Numbers-6.png)

### Even and Odd Numbers

For all the integers, it is true that either they are divisible by 2 or they are not divisible by 2. Those numbers that are divisible by 2 are called even numbers, and those that are not divisible by 2 are called odd numbers. The image added below shows the Even Numbers.

![Even Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230505075956/Numbers-3.png)

The image added below shows the Odd Numbers.

![Odd Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230505080003/Numbers-4.png)

Examples of Even Numbers are 0, 2, 4, 6, . . . and so on, as well as negative numbers, i.e., -2, -4, -6, . . ., and so on. Examples of Odd Numbers are 1, 3, 5, 7, . . ., and so on, as well as negative numbers, i.e., -1, -3, -5, -7, . . . and so on.

### Prime and Composite Numbers

Natural Numbers that are divisible by either 1 or themselves are known as [Prime Numbers](https://www.geeksforgeeks.org/maths/prime-numbers/) and if there are any other divisors of a number other than 1 and itself, then it is called a [composite number](https://www.geeksforgeeks.org/maths/composite-numbers/).

The smallest prime number is 2, and some other examples of Prime Numbers are 3, 5, 7, 11, etc. Also, some examples of composite numbers are 4, 6, 8, 9, and many, many more.

> ****Note:**** 1 is neither Prime nor Composite Number.

The following image shows the prime numbers.

![Prime-Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20240307162409/Prime-Numbers.png)

****Also Check:****

> - [Composite Numbers Interesting Facts](https://www.geeksforgeeks.org/maths/composite-numbers-interesting-facts/)
> - [Interesting Facts about Prime Numbers](https://www.geeksforgeeks.org/maths/amazing-facts-about-prime-numbers/)

### Coprime Numbers

In mathematics, two numbers a and b (which do not need to be prime) are called [Coprime Numbers](https://www.geeksforgeeks.org/maths/co-prime-numbers/), relatively prime or mutually prime, if and only if they only have 1 as their common factor.

For example, 4 = 22 and 9 = 32 both only have 1 as their common factor, or we can also say that the HCF of 4 and 9 is 1. So, 4 and 9 are Coprime Numbers.

Some more examples of coprime pairs are 12 and 25, 8 and 15, 24 and 35, etc. The following image shows how 21 and 22 are co-prime numbers.

![Coprime Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230505080041/Numbers-9.png)

### Perfect Numbers

Perfect numbers are those natural numbers whose sum of divisors excluding themselves is equal to the number itself, i.e., if we calculate the sum of divisors of a number excluding itself and it comes out to be the same as the original number, then that number is called a Perfect Numbers.

For example, consider 6, whose divisors are 1, 2, and 3, and whose sum is 6. Thus, 6 is the Perfect Number and its image is added below,

![Perfect Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230801190129/Numbers.png)

Some other examples are 28, 496, 8128, etc.

### Algebraic and Transcendental Numbers

All the numbers mentioned above are examples of Algebraic Numbers that are defined as the solutions of some algebraic equations, i.e., Algebraic Numbers are those numbers that are the solutions of some algebraic equations.

> ****For example****, the solution of x2 + 1 = 0 is ±√-1, which is an algebraic number. Some other examples are all the real numbers as well as complex numbers.

On the other hand, all such numbers that can't be found as a solution to some algebraic equation are called Transcendental Numbers. Some of the most famous examples of Transcendental Numbers are [****π (pi)****](https://www.geeksforgeeks.org/maths/value-of-pi/) and [****e (Euler's number)****](https://www.geeksforgeeks.org/maths/value-of-e/)

## Prime Factorization

Expressing any composite number as the product of prime numbers is called [****prime factorization****](https://www.geeksforgeeks.org/maths/prime-factorization/). The prime factorization of a number "x" can be found easily by dividing x by the smallest possible prime number and then repeating the process until the quotient is either a prime number or 1.

For example, prime factorization of 12 = 2 × 2 × 3 or 22 × 3. Some more examples are 15 = 5 × 3, 33 = 3 × 11, 42 = 2 × 3 × 7, etc. The prime factor of 18 is discussed in the image below.

![prime-factorization](https://media.geeksforgeeks.org/wp-content/uploads/20250408154557518288/prime-factorization.webp)

Prime Factorization of 18

> ****Check:**** [Interesting Facts about Prime Factorization](https://www.geeksforgeeks.org/maths/interesting-facts-about-prime-factorization/)

## HCF

[HCF (Highest Common Factor)](https://www.geeksforgeeks.org/maths/greatest-common-divisor-gcd/) is the largest possible number that can divide two or more numbers without leaving any remainder. HCF is also called the Greatest Common Divisor. For example, 6 is the HCF of 12 and 18, 12 is the HCF of 12 and 24, 7 is the HCF of 14 and 21, etc. The image discussed below shows the HCF of 12 and 18.

![HCF](https://media.geeksforgeeks.org/wp-content/uploads/20230505080105/Numbers-14.png)

## LCM

[LCM (Lowest Common Multiple)](https://www.geeksforgeeks.org/maths/lcm-least-common-multiple/) of any two or more numbers is the smallest possible number when divisible by all the given numbers, yields a remainder of 0. In other words, the LCM of any two or more numbers is the smallest common multiple of all numbers. For example, the LCM of 12 and 14 is 84, LCM(21, 24) = 168, LCM(42, 122) = 2562, etc. The image discussed below shows the LCM of 4 and 6.

![LCM](https://media.geeksforgeeks.org/wp-content/uploads/20230505130425/Numbers-15-(1).webp)

> ****Read more about:**** [****HCF and LCM****](https://www.geeksforgeeks.org/maths/hcf-and-lcm/)****.****

## Number System

The Number System is the set of guidelines that gives meaning to expressions written in that number system. For example, if we want to express that we have ten dogs, in the decimal number system we would write "10 dogs," in the binary system "1010 dogs," in the octal system "12 dogs," and in the hexadecimal system "A dogs." All these statements represent ten dogs but in different number systems. 

Any Number System needs two things to express all the numbers we want it to represent. First are the symbols (generally all number systems that need less than or equal to 10 symbols use modern-day decimal numerals), and the second is the base (which is the number of required symbols).

****For example****, in the decimal number system, there are ten symbols, so its base is 10.

> ****Check:**** [****Number System in Maths****](https://www.geeksforgeeks.org/maths/number-system-in-maths/)

### Decimal Number System

The [Decimal Number System](https://www.geeksforgeeks.org/digital-logic/decimal-number-system/) is the most used number system out of all. There are 10 digits in this number system, which are 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9. The decimal numbers are represented as,

![Decimal Number System](https://media.geeksforgeeks.org/wp-content/uploads/20230505080052/Numbers-11.png)

### Binary Number System

In the [Binary Number System](https://www.geeksforgeeks.org/maths/binary-number-system/), there are only two digits, and using those, we express all the numbers. The most common numerals for the Binary System are 0 and 1, but we can use any pair of symbols to represent the same as long as the symbols are well-defined.

> ****For example****, 10010001, 11011001, and 1010 are some examples of binary numbers which in decimals represent 145, 217, and 10 respectively.

For a Better understanding of the [conversion of binary to decimal](https://www.geeksforgeeks.org/utilities/binary-to-decimal/) read this article. In the binary system, we use two bits 0 and 1 as shown in the image below,

![Binary Number System](https://media.geeksforgeeks.org/wp-content/uploads/20230512125120/Numbers-2-6.webp)

### Octal Number System

In the [Octal Number System](https://www.geeksforgeeks.org/maths/octal-number-system/), there are only 8 digits or symbols which are generally represented with modern-day decimal symbols by only up to 7 i.e., 0, 1, 2, 3, 4, 5, 6, 7. Using these 8 symbols we can write and express all the numbers.

> ****For example****, 231, 41, and 653 are some examples of octal numbers which in decimals represent 153, 33, and 427 respectively. The digits used in the Octal Number system are shown in the image below,

![Octal Number System](https://media.geeksforgeeks.org/wp-content/uploads/20230512125119/Numbers-2-7.webp)

### Hexadecimal Number System

In the [Hexadecimal Number System](https://www.geeksforgeeks.org/maths/hexadecimal-number-system/), there are 16 numerals: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, and F. It is widely used in programming-related tasks as its base is 16 (which is a multiple of 2), which is the foundation of computing as every bit can only have two values: on or off.

Some examples of the Hexadecimal Number System are A1, 2A, and F3, which in the decimal number system represent 161, 42, and 243, respectively.

## Properties of Numbers

There are some properties that different types of numbers have with different defined operations such as addition or multiplication, and those properties are as follows,

- [Commutative Property](https://www.geeksforgeeks.org/maths/commutative-property/)
- [Associative Property](https://www.geeksforgeeks.org/maths/associative-property/)
- [Identity Property](https://www.geeksforgeeks.org/maths/identity-property/)
- Inverse Property
- [Distributive Property](https://www.geeksforgeeks.org/maths/distributive-property/)

### Commutative Property

For any two numbers a and b, for any operation ×, if a×b=b×a, then × is commutative for those sets of numbers.

> ****For example****, addition and multiplication of all complex numbers hold the commutative property but with subtraction and division, they don't hold commutative property.

### Associative Property

For any three numbers a, b, and c, and any operation ×, if a×(b×c)=(a×b)×c, then × operations hold the associative property for those sets of numbers.

> ****For example****, addition and multiplication of all complex numbers hold the associative property but with subtraction and division, they don't hold associative property.

### Identity Property

Identity is the uniquely defined number with respect to an operation such that operating identity to any number results in the number itself i.e., for operation ×, a × e = a (where e is the identity w.r.t operation ×).

> ****For example****, 0 is the identity for the addition operation as a + 0 = a and 1 is the identity of multiplication as a × 1 = a.

### Inverse Property

The Inverse is the uniquely defined number for each number with respect to some operation, such that when operating any number with its inverse, the output is an identity for that operation. In other words, for some number a and operation ×, where e is the identity, if a×b=e then b is called the inverse of a with respect to operation ×.

> ****For example****, -1 is the inverse of 1 under addition as 1+(-1) = 0 (0 is the identity of addition), and 1/2 is the inverse of 2 under multiplication as 1/2×2 = 1(1 is the identity of multiplication).

### Distributive Property

Distributive Property is defined for a pair of operations, say o and ×. For some numbers a, b, and c if ao(b×c) = aob×aoc then o distributes over ×, and if a × (boc) = a × boa × c then × distributes over o.

> ****For example****, multiplication distributes over addition but addition doesn't distribute over multiplication.

****Check:**** [****Number Theory for DSA & Competitive Programming****](https://www.geeksforgeeks.org/competitive-programming/number-theory-competitive-programming/)

## History of Numbers

The early brain of humans was capable of grasping the concept of numbers, such that they could see how many cattle they owned or how much food would suffice for the community, but the present-day concept of numbers and counting is foreign to them. It was believed by scientists that the idea of numbers and counting originally originated in ancient societies such as Egypt, Mesopotamia, and India.

![different numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230228160116/Numbers-1.png)

During the evolution of humans, a lot of different numerical systems were developed by humans in different parts of the world, and as the world became more connected, these systems traveled with people from one continent to another. One such numerical system, i.e., the Indian numeral system, travelled to countries in the Middle East and Europe and laid the foundation stone for the modern-day decimal system.

![Different numerals from the history](https://media.geeksforgeeks.org/wp-content/uploads/20230228160034/Numbers-3.png)

After reaching the Middle East, Arabic academics and scholars made noteworthy contributions to the growth of mathematics during the Middle Ages, including the usage of algebra and the decimal point. Mathematicians like John Napier and Simon Stevin introduced the ideas of decimal fractions and logarithms, respectively, in the 16th century, which helped to simplify complicated computations.

With new findings and applications in disciplines like physics, engineering, and computer science, the area of mathematics has advanced in the modern age. Today, numbers and mathematical ideas are fundamental to many key aspects of our everyday lives, from time and distance measurement to money management and data analysis.

## Solved Examples on What are Numbers

****Example 1: Find the LCM of 12 and 18.****

****Answer:****

> Multiples of 12 are 12, 24, 36, 48, 60, 72, ...  
> Multiples of 18 are 18, 36, 54, 72, ...
> 
> We can see that 36 is the smallest number that is divisible by both 12 and 18. 
> 
> Hence, the LCM of 12 and 18 is 36.

****Example 2: Find the HCF of 24 and 36.****

****Answer:****

> Factors of 24 are: 1, 2, 3, 4, 6, 8, 12, 24  
> Factors of 36 are: 1, 2, 3, 4, 6, 9, 12, 18, 36
> 
> We can see that 12 is the largest number that is common to both 24 and 36. 
> 
> Hence, the HCF of 24 and 36 is 12.

****Example 3: Find the prime factorization of 72.****

****Answer:****

> We can start by dividing 72 by its smallest prime factor, which is 2.  
> 72 ÷ 2 = 36
> 
> Now, we can divide 36 by its smallest prime factor, which is 2.  
> 36 ÷ 2 = 18
> 
> Again, we can divide 18 by its smallest prime factor, which is 2.  
> 18 ÷ 2 = 9
> 
> Again, we can divide 9 by its smallest prime factor, which is 3.  
> 9÷ 3 = 3
> 
> 3 is a prime number, so we can stop here.
> 
> Therefore, the prime factorization of 72 is 2 × 2 × 2 × 3 × 3.

****Example 4: Write 2341, 3247, 981472 in words (in Indian System).****

****Answer:****

> - 2,341: "Two Thousand Three Hundred Forty-One"
> - 3,247: "Three Thousand Two Hundred Forty-Seven"
> - 9,81,472: "Nine Lac  Eighty-One Thousand Four Hundred Seventy Two" 

## Practice Questions on Numbers Unsolved

****Question 1:**** Find the LCM of 20 and 30.

****Question 2:**** Find the HCF of 45 and 60.

****Question 3:**** Find the prime factorization of 84.

****Question 4:**** Write 4567, 8290, and 678912 in words.

****Question 5:**** Find the LCM of 8 and 12.

****Question 6:**** Find the HCF of 36 and 54.

****Question 7:**** Find the prime factorization of 144.

****Question 8:**** Write 1234, 56789, and 100000 in words.

  

Numbers

---------------


[Numbers](https://www.geeksforgeeks.org/maths/numbers/) are used in various arithmetic values to carry out various arithmetic operations such as addition, subtraction, multiplication, and so on that are utilized in daily life for the purpose of computation. The value of a number is defined by the digit, its place value in the number, and the number system's base. Numbers, often known as numerals, are mathematical values used for counting, measuring, labeling, and quantifying fundamental quantities.

[Numbers](https://www.geeksforgeeks.org/maths/numbers/) are mathematical values or numbers used to measure or calculate quantities. It is denoted by numerals such as 2, 4, 7, and so on. Integers, whole numbers, natural numbers, rational and irrational numbers, and so on are all examples of numbers. Counting is the process of expressing the number of components or objects that are given. 

### Counting numbers

Counting numbers are natural numbers that can be numbered and are always positive. Counting is necessary for everyday life since we need to count the number of hours, days, money, and so on. [Numbers](https://www.geeksforgeeks.org/maths/numbers/) can be counted and written in words such as one, two, three, four, and so on.

Counting numbers are natural numbers that can be counted. They begin with 1 and progress through the series as 1, 2, 3, 4, and so on. We cannot count 0 because it is not included in the counting of numbers.

### How many counting numbers are there?

****Answer:**** 

> Counting numbers are simply natural numbers. Counting numbers, often known as natural numbers, are positive numbers that range from one to infinity. 'N' represents the set of natural numbers. It is the number system that we are most associated with. N = 1, 2, 3, 4, 5, 6, 7 represents the set of natural numbers. 
> 
> Natural numbers are sometimes known as counting numbers since they can be counted with one's hands. The natural numbers are a subset of the number system that consists of positive integers starting with 1. These figures are shown on the right side of a number line. Natural numbers include 1, 2, 3, 4, 5,... N. Natural numbers do not include zero and negative integers.
> 
> The lowest natural number that can exist is one, while the greatest natural number that may occur is any infinite positive integer value. Natural numbers do not include fractions, decimal values, complex numbers, and so on.
> 
> Counting numbers is defined as the set of numbers we used to count things. Natural numbers are numbers that can be counted. And the figures are always in the positive.
> 
> ****There are infinite counting numbers and this is in infinite value numbers, Therefore, which makes impossible to tell exact numbers as these are uncountable.****

### Sample Questions 

****Question 1: Identity from below numbers which are counting numbers?****

****5, 0, 1.008, 5/7, 2, 96****

****Answer:**** 

> Counting numbers are sometimes known as counting numbers since they can be counted with one's hands. The natural numbers are a subset of the number system that consists of positive integers starting with 1
> 
> Here 5, 2 , 96 are the counting numbers.

****Question 2: Is 17 a counting number or integer?****

****Answer:**** 

> Counting numbers are the positive numbers that count from 1 to infinity. Therefore, 17 being a positive number is a counting number as well as positive integer.

****Question 3: What are the smallest two-digit and largest two-digit counting numbers?****

****Answer:****

> The set of numbers that we used to count anything is defined as counting numbers. All-natural numbers are counting numbers. And these numbers are always positive.
> 
> - Smallest two digit counting number is 10
> - Largest two digit counting number is 99

****Question 4: From these below numbers, which are integer as well as counting numbers?****

****8/4, 4.59, 45454/123, -2, -5, -88,6, 33, 455****

****Answer:****

> Here, Integers are: -2, -5, -88, 6, 33, 455 only as integers does not include fractional or decimal value. But as we know all integers are not counting numbers. So here from above integers only which are counting numbers are 6 , 33, 455

****Question 5: What are the Odd Counting Numbers Between 5 and 20?**** 

****Answer:****  

> Odd numbers are the numbers which are not divisible by 2. So, the odd counting numbers between 5 and 16 are 7, 9, 11, 13, 15 , 17 , 19

****Question 6: What are the Even Counting Numbers Between 5 and 25?**** 

****Answer:**** 

> Even numbers are the numbers that are divisible by 2. So the even counting numbers between 5 and 25 are 6,8,10,12,14,16,18,20,22,24

----------------

[****How many Counting Numbers are there?****](https://www.geeksforgeeks.org/maths/how-many-counting-numbers-are-there/)
- [****What is another name for Natural Numbers?****](https://www.geeksforgeeks.org/maths/what-is-another-name-for-natural-numbers/)
- [****Is 0 a Natural Number?****](https://www.geeksforgeeks.org/maths/is-0-a-natural-number/)

## Counting Numbers Worksheet

****Question 1: Complete the following worksheet by counting the numbers by 5.****

- ****5, ?, 15, 20, ?****
- ****25, 30, ?, 40****
- ****1000, ?, 1010, 1015, 1020****

****Answer:****

> As the numbers are counted by 5, adding 5 to the previous number will give the next number. Therefore, the answers are:
> 
> - 5, ****10****, 15, 20, ****25****
> - 25, 30, ****35****, 40
> - 1000, ****1005****, 1010, 1015, 1020

****Question 2: Write the counting numbers from:****

- ****10 to 20****
- ****100 to 1000 counting by 100****
- ****25 to 50****

****Answer:****

> - Counting numbers from 10 to 20 ⇢ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.
> - Counting numbers from 100 to 1000 by 100 ⇢ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000.
> - Counting numbers from 25 to 50 ⇢ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50.

****Question 3: Write 11 counting numbers from 9.****

****Answer:****

> 11 counting numbers from 9 are 9, 10, 11, 12, 13, 14, 15, 16, 17.

****Question 4: Write only the even counting numbers from 1 to 20.****

****Answer:****

> Even counting numbers will only include even numbers. Even numbers are the number that are completely divisible by 2 leaving 0 as a remainder. So, even counting numbers from 1 to 20 are:
> 
> 2, 4, 6, 8, 10, 12, 14, 16, 18, 20

****Question 5: Write the odd counting numbers from 10 to 50.****

****Answer:****

> Odd counting numbers will only include odd numbers. Odd numbers are the number that are not completely divisible by 2. So, odd counting numbers from 10 to 50 are:
> 
> 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49.

- Natural numbers or [counting numbers](https://www.geeksforgeeks.org/maths/counting-numbers/) are those integers that begin with 1 and go up to infinity.
- Only positive integers, such as 1, 2, 3, 4, 5, 6, etc., are included in the set of natural numbers. ****Natural numbers start from**** 1 and go up to ∞.
- Natural numbers are the set of positive integers starting from 1 and increasing incrementally by 1.
- They are used for counting and ordering.
- The set of natural numbers is typically denoted by ****N**** and can be written as {1, 2, 3, 4, 5, …}

> ****Check:**** [Is Zero a Natural Number?](https://write.geeksforgeeks.org/preview/is-0-a-natural-number/)

Table of Content

- [Characteristics of Natural Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/#key-characteristics)
- [Types of Natural Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/#types-of-natural-numbers)
- [Natural Numbers from 1 to 100](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/#natural-numbers-from-1-to-100)
- [Natural Numbers and Whole Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/#natural-numbers-and-whole-numbers)
- [Properties of Natural Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/#properties-of-natural-numbers)
- [Operations With Natural Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/#operations-with-natural-numbers)
- [Solved Examples of Natural Numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/#examples-of-natural-numbers)

## Characteristics of Natural Numbers

****No Decimals or Fractions****: One defining characteristic of natural numbers is that they are whole numbers. Unlike rational or real numbers, natural numbers do not include decimals or fractions. For example, numbers like 3.14 or 1/2 are not natural numbers​.

****Starting from 1****: Natural numbers always start from 1 and go up to infinity. This distinguishes them from whole numbers, which include zero​.

****Non-Negative****: By definition, natural numbers are always positive integers, which means they don’t include negative numbers like -1, -2, etc.​

### Set of Natural Numbers

In mathematics, the set of natural numbers is expressed as 1, 2, 3, ... The set of natural numbers is represented by the symbol N. N = {1, 2, 3, 4, 5, ... ∞}. A collection of elements is referred to as a set (numbers in this context). The smallest element in N is 1, and the next element in terms of 1 and N for any element in N. 2 is 1 greater than 1, 3 is 1 greater than 2, and so on. The below table explains the different [set forms](https://www.geeksforgeeks.org/maths/representation-of-a-set/) of natural numbers.

|Set Form|Explanation|
|---|---|
|Statement Form|N = Set of numbers generated from 1.|
|Roaster Form|N = {1, 2, 3, 4, 5, 6, ...}|
|Set-builder Form|N = {x: x is a positive integer starting from 1}|

Natural numbers are the subset of whole numbers, and whole numbers are the subset of integers. Similarly, integers are the subset of real numbers.

## Types of Natural Numbers

### Odd natural numbers

Odd natural numbers are integers greater than zero that cannot be divided evenly by 2, resulting in a remainder of 1 when divided by 2. Examples of odd natural numbers include 1, 3, 5, 7, 9, 11, and so on.

### Even natural numbers

Even natural numbers are whole numbers that are divisible by 2 without leaving a remainder. In other words, they are integers greater than zero that can be expressed in the form 2n, where n is an integer. Examples of even natural numbers include 2, 4, 6, 8, 10, and so on.

## Natural Numbers from 1 to 100

As Natural Numbers are also called counting numbers, thus natural numbers from 1 to 100 are:

> ****1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100****.

## ****Natural Numbers and Whole Numbers****

The set of whole numbers is identical to the set of natural numbers, with the exception that it includes a 0 as an extra number.

> ****W = {0, 1, 2, 3, 4, 5, ...}**** and ****N = {1, 2, 3, 4, 5, ...}****

![Natural Numbers and Whole Numbers](https://media.geeksforgeeks.org/wp-content/uploads/20230608113319/Natural-Numbers.png)

****Read more about****: [Whole Number](https://www.geeksforgeeks.org/maths/whole-numbers/)

## Difference Between Natural Numbers and Whole Numbers

Let's discuss the differences between natural numbers and whole numbers.

|****Natural Numbers Vs Whole Numbers****|   |
|---|---|
|Natural Numbers|Whole Numbers|
|---|---|
|The smallest natural number is 1.|The smallest whole number is 0.|
|All natural numbers are whole numbers.|All whole numbers are not natural numbers.|
|Representation of the set of natural numbers is N = {1, 2, 3, 4, ...}|Representation of the set of whole numbers is W = {0, 1, 2, 3, ...}|

## ****Natural Numbers on Number Line****

Natural numbers are represented by all positive integers or integers on the right-hand side of 0, whereas whole numbers are represented by all positive integers plus zero.

Here is how we represent natural numbers and whole numbers on the number line:

![Natural Numbers on Number Line](https://media.geeksforgeeks.org/wp-content/uploads/20220830182758/Naturalnumbersonnumberline.jpg)

Representation of Natural Numbers on Number Line

## ****Properties of Natural Numbers****

All the natural numbers have these properties in common :

1. [Closure property](https://www.geeksforgeeks.org/maths/closure-property/)
2. [Commutative property](https://www.geeksforgeeks.org/maths/commutative-property/)
3. [Associative property](https://www.geeksforgeeks.org/maths/associative-property/)
4. [Distributive property](https://www.geeksforgeeks.org/maths/distributive-property/)

Let's learn about these properties in the table below.

|****Property****|****Description****|****Example****|
|---|---|---|
|****Closure Property****|   |   |
|Addition Closure|The sum of any two natural numbers is a natural number.|3 + 2 = 5, 9 + 8 = 17|
|Multiplication Closure|The product of any two natural numbers is a natural number.|2 × 4 = 8, 7 × 8 = 56|
|****Associative Property****|   |   |
|Associative Property of Addition|Grouping of numbers does not change the sum.|1 + (3 + 5) = 9, (1 + 3) + 5 = 9|
|Associative Property of Multiplication|Grouping of numbers does not change the product.|2 × (2 × 1) = 4, (2 × 2) × 1 = 4|
|****Commutative Property****|   |   |
|Commutative Property of Addition|The order of numbers does not change the sum.|4 + 5 = 9, 5 + 4 = 9|
|Commutative Property of Multiplication|The order of numbers does not change the product.|3 × 2 = 6, 2 × 3 = 6|
|****Distributive Property****|   |   |
|Multiplication over Addition|Distributing multiplication over addition.|a(b + c) = ab + ac|
|Multiplication over Subtraction|Distributing multiplication over subtraction.|a(b - c) = ab - ac|

> ****Note:****
> 
> - Subtraction and Division may not result in a natural number.
> - Associative Property does not hold true for subtraction and division.

## Operations With Natural Numbers

We can add, subtract, multiply, and divide the natural numbers together but the result in the subtraction and division is not always a natural number.

Let's understand the operations on natural numbers:

|Operation|Description|Symbol|Examples|
|---|---|---|---|
|****Addition****|Combines two or more numbers to find their total.|+|3 + 4 = 7, 11 + 17 = 28|
|****Subtraction****|Finds the difference between two natural numbers; can result in natural or non-natural numbers.|-|5 - 3 = 2, 17 - 21 = -4|
|****Multiplication****|Finds the value of repeated addition.|× or *|3 × 4 = 12, 7 × 11 = 77|
|****Division****|Dividing the number into equal parts; may result in a quotient and a remainder.|÷ or /|12 ÷ 3 = 4, 22 ÷ 11 = 2|
|****Exponentiation****|Raises a number to a certain power.|^|23 = 8|
|****Square Root****|The value that, when multiplied by itself, gives the original number.|√|√25 = 5|
|****Factorial****|The product of all positive integers up to and including that number.|!|5! = 120|

## Mean of First n Natural Numbers

As mean is defined as the ratio of the sum of observations to the number of total observations.

[Mean Formula](https://www.geeksforgeeks.org/dsa/mean/) for the first ****n**** terms of natural number:

> ****Mean = S/n = (n+1)/2****

where,

- ****S**** is the Sum of all Observations
- ****n**** is the Number of Terms Taken into Consideration

## Sum of Square of First n Natural Numbers

The sum of the square of the first n natural numbers is given as follows:

> ****S = n(n + 1)(2n + 1)/6****

Where ****n**** is the number taken into consideration.

****People Also Read:****

> - [Number System](https://www.geeksforgeeks.org/maths/number-system-in-maths/)
> - [Real Numbers](https://www.geeksforgeeks.org/maths/real-numbers/)
> - [Rational Numbers](https://www.geeksforgeeks.org/maths/rational-numbers/)
> - [Another Name for Natural Numbers](https://www.geeksforgeeks.org/maths/what-is-another-name-for-natural-numbers/)

## Solved Examples of Natural Numbers

Let's solve some example problems on Natural Numbers.

****Question 1: Identify the natural numbers among the given numbers: 23, 98, 0, -98, 12.7, 11/7, 3.****

****Solution:****

> Since negative numbers, 0, decimals, and fractions are not a part of natural numbers.  
> Therefore, 0, -98, 12.7, and 11/7 are not natural numbers.  
> Thus, natural numbers are 23, 98, and 3.

****Question 2: Prove the distributive law of multiplication over addition with an example.****

****Solution:****

> Distributive law of multiplication over addition states: a(b + c) = ab + ac
> 
> For example, 4(10 + 20), here 4, 10, and 20 are all natural numbers and hence must follow distributive law  
> 4(10 + 20) = 4 × 10 + 4 × 20  
> 4 × 30 = 40 + 80  
> 120 = 120  
> Hence, proved.

****Question 3: Prove the distributive law of multiplication over subtraction with an example.****

****Solution:****

> Distributive law of multiplication over addition states: a(b - c) = ab - ac.
> 
> For example, 7(3 - 6), here 7, 3, and 6 are all natural numbers and hence must follow the distributive law. Therefore,  
> 7(3 - 6) = 7 × 3 - 7 × 6  
> 7 × -3 = 21 + 42  
> -21 = -21  
> Hence, proved.

****Question 4: List the first 10 natural numbers.****

****Solution:****

> 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10 are the first ten natural numbers.

## Natural Numbers Question

****Question 1:**** What is the Smallest Natural Number?

****Question 2:**** What is the Biggest Natural Number?

****Question 3:**** Simplify, 17(13 - 16)

****Question 4:**** Simplify, 11(9 - 2)

****Question 5:**** Find the sum of the first 20 natural numbers.

****Question 6:**** Is 97 a prime natural number?

****Question 7:**** What is the smallest natural number that is divisible by both 12 and 18?

****Question 8:**** Find the product of the first 5 natural numbers.

****Question 9:**** How many natural numbers are there between 50 and 100 (inclusive)?

****Question 10:**** Discuss whether 0 is included in the set of natural numbers based on its definition.

****Answer Key:****

> 1. ****1****
> 2. ****Not defined****
> 3. ****-51****
> 4. ****77****
> 5. ****210****
> 6. ****Yes****
> 7. ****36****
> 8. ****120****
> 9. ****51****
> 10. ****No****

## Conclusion

- Natural numbers form the foundation of the number system, containing all positive integers from 1 to infinity.
- They are important for counting and ordering, playing a crucial role in everyday mathematics and various advanced fields.
- Natural numbers exhibit properties like closure (the sum or product of two natural numbers is also a natural number), commutative, associative, and distributive properties.
- Basic operations with natural numbers include addition, subtraction, multiplication, division, exponentiation, square roots, and factorials.


If a set is constructed using all-[natural](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/) [numbers](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/), zero, and negative natural numbers, then that set is referred to as Integer. Integers range from negative infinity to positive infinity.

- ****Natural Numbers:**** [Numbers](https://www.geeksforgeeks.org/maths/numbers/) greater than zero are called positive numbers. ****Example:**** 1, 2, 3, 4...
- ****Negative of Natural Numbers:**** Numbers less than zero are called negative numbers. ****Example:**** -1, -2, -3, -4...
- ****Zero (0)**** is neither positive nor negative.

### Symbol of Integers

Set of integers is represented by the letter Z as shown below:

****Z = {... -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7...}****

## Types of Integers

Integers are classified into three categories:

- ****Zero (0)****
- ****Positive Integers (i.e. Natural numbers)****
- ****Negative Integers (i.e. Additive inverses of Natural Numbers)****

### ![Classification-of-Integers](https://media.geeksforgeeks.org/wp-content/uploads/20230814153648/Classification-of-Integers.png)Zero

Zero is a unique number that does not belong to the category of positive or negative integers. It is considered a neutral number and is represented as "0" without any plus or minus sign.

### Positive Integers

Positive integers, also known as natural numbers or counting numbers, are often represented as Z+. Positioned to the right of zero on the number line, these integers encompass the realm of numbers greater than zero.

> ****Z********+**** ****→**** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,….

### Negative Integers

Negative integers mirror the values of natural numbers but with opposing signs. They are symbolized as Z–. Positioned to the left of zero on the number line, these integers form a collection of numbers less than zero.

> ****Z********–**** ****→**** -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30,…..

## Integers on a Number Line

As we have discussed previously, it is possible to visually represent the three categories of integers - positive, negative, and zero - on a number line.

Zero serves as the midpoint for ****integers on the number line****. Positive integers occupy the right side of zero, while negative integers populate the left side. Refer to the diagram below for a visual representation.

![Integers on Number Line](https://media.geeksforgeeks.org/wp-content/uploads/20230814153930/Integers-on-Number-Line.png)

## ****Rules of Integers****

Various rules of integers are,

- ****Addition of Positive Integers****: When two positive integers are added together, the result is always an integer.
- ****Addition of Negative Integers****: Sum of two negative integers results in an integer.
- ****Multiplication of Positive Integers****: Product of two positive integers yields an integer.
- ****Multiplication of Negative Integers****: When two negative integers are multiplied, the outcome is an integer.
- ****Sum of an Integer and Its Inverse****: Sum of integer and its inverse is alays zero.
- ****Product of an Integer and Its Reciprocal****: Product of an Integer and Its Reciprocal is always 1.

## Arithmetic Operations on Integers

Four basic Maths operations performed on integers are:

- [Addition](https://www.geeksforgeeks.org/maths/addition/#:~:text=) of Integers
- [Subtraction](https://www.geeksforgeeks.org/maths/subtraction/#:~:text=) of Integers
- [Multiplication](https://www.geeksforgeeks.org/maths/multiplication/#:~:text=) of Integers
- [Division](https://www.geeksforgeeks.org/maths/division/) of Integers

### Addition of Integers

Addition of [integers](https://www.geeksforgeeks.org/maths/integers/) is similar to finding the sum of two integers. Read the rules discussed below to find the sum of integers.

****Example: Add the given integers****

- ****3 + (-9)****

- ****(-5) + (-11)****

> - 3 + (-9) = -6
> 
> - (-5) + (-11) = -16

### Subtraction of Integers

Subtraction of integers is similar to finding the difference between two integers. Read the rules discussed below to find the difference between integers.

****Example: Add the given integers****

- ****3 - (-9)****

- ****(-5) - (-11)****

> - 3 - (-9) = 3 + 9 = 12
> 
> - (-5) - (-11) = -5 + 11 = 6

### Multiplication of Integers

Multiplication of integers is achieved by following the rule:

- When both integers have same sign, the product is positive.

- When both integers have different signs, the product is negative.

|Product of Sign|Resultant Sign|Example|
|---|---|---|
|(+) × (+)|+|9 × 3 = 27|
|(+) × (–)|-|9 × (-3) = -27|
|(–) × (+)|-|(-9) × 3 = -27|
|(–) × (–)|+|(-9) × (-3) = 27|

### Division of Integers

Division of integers is achieved by following the rule:

- When both integers have the same sign, the division is positive.
- When both integers have different signs, the division is negative.

|Division of Sign|Resultant Sign|Example|
|---|---|---|
|(+) ÷ (+)|+|9 ÷ 3 = 3|
|(+) ÷ (–)|-|9 ÷ (-3) = -3|
|(–) ÷ (+)|-|(-9) ÷ 3 = -3|
|(–) ÷ (–)|+|(-9) ÷ (-3) = 3|

## Properties of Integers

Integers have various properties, the major properties of integers are:

- ****Closure Property****
- ****Associative Property****
- ****Commutative Property****
- ****Distributive Property****
- ****Identity Property****
- ****Additive Inverse****
- ****Multiplicative Inverse****

### Closure Property

[Closure property](https://www.geeksforgeeks.org/maths/closure-property/) of integers states that if two integers are added or multiplied together their result is always an integer. For integers p and q

- p + q = integer
- p × q = integer

****Example:****

> (-8) + 11 = 3 (An integer)  
> (-8) × 11 = -88 (An integer)

### Commutative Property

[Commutative property](https://www.geeksforgeeks.org/maths/commutative-property/) of integers states that for two integers p and q

- p + q = q + p
- p × q = q × p

****Example:****

> (-8) + 11 = 11 + (-8) = 3  
> (-8) × 11 = 11 × (-8) = -88

But the commutative property is not applicable to the subtraction and division of integers.

### Associative Property

[Associative property](https://www.geeksforgeeks.org/maths/associative-property/) of integers states that for integers p, q, and r

- p + (q + r) = (p + q) + r
- p × (q × r) = (p × q) × r

****Example:****

> 5 + (4 + 3) = (5 + 4) + 3 = 12  
> 5 × (4 × 3) = (5 × 4) × 3 = 60

### Distributive Property

[Distributive property](https://www.geeksforgeeks.org/maths/distributive-property/) of integers states that for integers p, q, and r

- p × (q + r) = p × q + p × r

****For Example, Prove: 5 × (9 + 6) = 5 × 9 + 5 × 6****

****Solution:****

> ****LHS**** = 5 × (9 + 6)   
>         = 5 × 15  
>         = 75
> 
> ****RHS**** = 5 × 9 + 5 × 6   
>         = 45 + 30  
>         = 75
> 
> Thus, LHS = RHS Proved

### Identity Property

Integers hold Identity elements both for addition and multiplication. Operation with the Identity element yields the same integers, such that

- p + 0 = p
- p × 1 = p

Here, 0 is Additive Identity, and 1 is Multiplicative Identity.

### Additive Inverse

Every integer has its [additive inverse.](https://www.geeksforgeeks.org/maths/additive-inverse-and-multiplicative-inverse/) An additive inverse is a number that in addition to the integer gives the additive identity. For integers, Additive Identity is 0. For example, take an integer p then its additive inverse is (-p) such that

- p + (-p) = 0

### Multiplicative Inverse

Every integer has its [multiplicative inverse](https://www.geeksforgeeks.org/maths/what-is-the-multiplicative-identity-and-multiplicative-inverse-of-the-complex-number/). A multiplicative inverse is a number that when multiplied to the integer gives the multiplicative identity. For integers, Multiplicative Identity is 1. For example, take an integer p then its multiplicative inverse is (1/p) such that

- p × (1/p) = 1

## Applications of Integers

****Integers**** extend beyond numbers, finding [applications of integers in real life](https://www.geeksforgeeks.org/maths/applications-of-integers-in-real-life/). Positive and negative values represent opposing situations. For instance, they indicate temperatures above and below zero. They facilitate comparisons, measurements, and quantification. ****Integers**** feature prominently in sports scores, ratings for movies and songs, and financial transactions like bank credits and debits.

### ****Articles Related to Integers:****

> - [Rational Number](https://www.geeksforgeeks.org/maths/rational-numbers/)
> - [Irrational Number](https://www.geeksforgeeks.org/maths/irrational-numbers/)
> - [Real Numbers](https://www.geeksforgeeks.org/maths/real-numbers/)
> - [Properties of Integers](https://www.geeksforgeeks.org/maths/properties-of-integers/)

## Integers Examples

Some examples on Integers are,

****Example 1: Can we say that 7 is both a whole number and a natural number?****

****Solution:****

> Yes, 7 is both whole number and natural number.

****Example 2: Is 5 a whole number and a natural number?****

****Solution:****  

> Yes, 5 is both a natural number and whole number.

****Example 3: Is 0.7 a whole number?****

****Solution:**** 

> No, it is a decimal.

****Example 4: Is -17 a whole number or a natural number?****

****Solution:**** 

> No, -17 is neither natural number nor whole number.

****Example 5: Categorize the given numbers among Integers, whole numbers, and natural numbers,****

- ****-3, 77, 34.99, 1, 100****

****Solution:****

> |Numbers|Integers|Whole Numbers|Natural Numbers|
> |---|---|---|---|
> |****-3****|Yes|No|No|
> |****77****|Yes|Yes|Yes|
> |****34.99****|No|No|No|
> |****1****|Yes|Yes|Yes|
> |****100****|Yes|Yes|Yes|

## Integers Class 6 Worksheet

Integers are a fundamental concept in mathematics, especially introduced at the class 6 level, aiming to broaden the understanding of numbers beyond natural numbers and whole numbers. Worksheet on Integers for students to solve is added below:

****Solve:****

1. 23 + (-12)
2. 15 – 12
3. -14 + 14
4. (13) × (-17)
5. (4) × (12)
6. 0 × (-87)
7. (114) ÷ (-7)
8. (-7) ÷ (-3)

### Answer Key:

1. 23 + (-12) = 11
2. 15 - 12 = 3
3. -14 + 14 = 0
4. 13 × (-17) = -221
5. 4 × 12 = 48
6. 0 × (-87) = 0
7. 114 ÷ (-7) = -16.29
8. -7 ÷ (-3) = 2.33

****Read More:**** [****Practice Questions on Integers****](https://www.geeksforgeeks.org/maths/practice-questions-on-integers/)

  

What are Integers?

[****Integers****](https://www.geeksforgeeks.org/maths/integers/)

![Closure-Property](https://media.geeksforgeeks.org/wp-content/uploads/20231226153058/Closure-Properyty.jpg)

Closure Property

Here, a and b are integers, and their sum, difference and multiplication all follow closure property for integers. But division of a and b is not necessarily a integer so division does not follow, closure property.

### Closure Property Definition

> When an arithmetic operation is performed to two numbers within a certain set of numbers, the final result always comes within the same set. Suppose a number set {5.10,15} is given. When you perform addition between 2 numbers of this set, the final number belongs to this set only.

## Types of Closure Property

As you have learned the definition of closure property, it's clear that arithmetic operations are performed within a number set. Now, we learn about the types of closure property.

Closure property is mainly divided into 4 parts:

- Closure Property of Addition
- Closure Property of Subtraction
- Closure Property of Multiplication
- Closure Property of Division

These four are the arithmetic operations we perform within a number set in closure property.

### Closure Property of Addition

As you can understand with the name itself, you perform addition within the set and the resultant will come under the set only. For example, let's take an even number set {2,4,6}. Take two numbers 2 & 4 from this set and perform addition on them. Here, 2+4= 6, the resultant is the same number as in the set.

Take another 2 numbers 4+6= 10, the resultant is not an even number. Here, this is violating the closure property.

|****Closure Property of Addition****|   |
|---|---|
|Real Numbers|a + b = Real number (a, b are real numbers.)|
|Rational Numbers|a + b = Rational number (a, b are real numbers.)|
|Integers Numbers|a + b = Integer (a, b are integers.)|
|Natural Numbers|a + b = Natural number (a, b are natural numbers)|
|Whole Numbers|a + b = Whole number (a, b are whole numbers)|

### Closure Property of Subtraction

Under this closure property, you perform subtraction within the set of numbers and the resultant will come under the set in the same way. For example, a number set {5,10,15} is given. Take 2 numbers 15 & 5 from this set and perform subtraction on them. Here, 15-5= 10, the outcome is under the set. Therefore it is following the closure property.

|****Closure Property of Subtraction****|   |
|---|---|
|Real Numbers|a - b = Real number (a, b are real numbers.)|
|Rational Numbers|a - b = Rational number (a, b are real numbers.)|
|Integers Numbers|a - b = Integer (a, b are integers.)|
|Natural Numbers|a - b = Natural number (a, b are natural numbers)|
|Whole Numbers|a - b = Whole number (a, b are whole numbers)|

### Closure Property of Multiplication

In closure under multiplication, you will multiply within the numbers in the set. Suppose take an even number set {2,4,8}. Multiply two numbers 2×4 = 8. It is an even number that's why following the closure property. While we perform multiplication 8×2 = 16, it's not an even number and not falling under the number set.

|****Closure Property of Multiplication****|   |
|---|---|
|Real Numbers|a × b = Real number (a, b are real numbers.)|
|Rational Numbers|a × b = Rational number (a, b are real numbers.)|
|Integers Numbers|a × b = Integer (a, b are integers.)|
|Natural Numbers|a × b = Natural number (a, b are natural numbers)|
|Whole Numbers|a × b = Whole number (a, b are whole numbers)|

### Closure Property of Division

For this type, let's take a set {8,16,24}. Take 2 numbers 16 & 8. Now we divide 16÷8= 2. Here, 2 is within the original number set. The closure property is satisfied here because 2 is falling under the same set where we perform division.

## Closure Property Formula

If we take two numbers a, and b from a set S then closure property formula states that, a (operator) b also belongs to set S. This is explained as,

> "****∀ a, b ∈ S ⇒ a (operator) b ∈ S****"

Real numbers are closed under Addition, Multiplication and Subtraction operation but not division operation because, a/b is not a real number when b is zero.

## Closure Property Examples

Now, you have understood what is closure property and what kind of arithmetic operations it includes. Now, it's time to deeply understand the concept of closure property via some real life examples. Below are some examples of closure property with various kind of number:

### Closure Property of Real Numbers

Real numbers also follows closure property. But first you need to understand what are real numbers.

Real numbers are the set of both rational numbers and irrational numbers. The real numbers are closed under all four arithmetic operations i.e. addition, subtraction, multiplication and division. When you perform any arithmetic operations on real numbers, the final result will also be a real number.

****Learn,**** [****Real Numbers****](https://www.geeksforgeeks.org/maths/real-numbers/)

Let's understand the closure property of real numbers with a simple example:

Suppose a number set {1, 2, 3} is given. Let's perform each arithmetic operation on this set to know whether it falls under closure property or not.

- ****Add**** 1 + 2 = 3, here 3 is also a real number.
- ****Subtract**** 3 - 2 = 1, here 1 is also a real number.
- ****Multiply**** 2 × 3 = 6, here 6 is also a real number.

- ****Divide**** 3 ÷ 0 = undefined {not a real number, and hence real number does not follow closure property under divide operator}

All the final results are real number and under the set. Therefore, closure property of real numbers is satisfied under Addition, Subtraction and Multiplication operation.

### Closure Property of Rational Numbers

Rational numbers also a follower of closure property. Now, the real question is what are rational numbers.

Rational numbers includes fraction and decimal numbers. When two rational numbers are added, subtracted or multiplied the resultant will always be a rational number. Here, the division is an exception because the division by zero is infinite. So the final answer will not always be a rational number while performing division.

****Learn,**** [****Rational Numbers****](https://www.geeksforgeeks.org/maths/rational-numbers/)

Let's take an example:

A number set {½, ⅔, ¾} is given.

- Add ½ + ⅔ = 7/6, which is also a rational number.
- Subtract ¾ - ½ = ¼. which is also a rational number.
- Multiply ½ × ⅔ = ⅓, which is also a rational number.
- Divide ¾ ÷ 0 = undefined {not a real number, and hence real number does not follow closure property under divide operator}.

### Closure Property of Integers

Integers also follow closure property under various arithmetic operations. Integers include both positive and negative numbers of a number scale. When you perform any arithmetic operations on integers, the final answer will always be an integer.

****Learn,**** [****Integers****](https://www.geeksforgeeks.org/maths/integers/)

Let's take a simple example:

A number set of integers {-3,0,3} is given.

- Add -3 + 3 = 0, which is an integer.
- Subtract 3 - 0= 3, which is an integer.
- Multiply 3 × 0 = 0, which is an integer.
- Divide -3 ÷ 0 = undefined {not a real number, and hence real number does not follow closure property under divide operator}.

### Closure Property in Modular Arithmetic

When you add, subtract, multiply or take reminders within a modulus, the result stays congruent to the set modulus. This property is called closure property in modular arithmetic.

## Closure Property in Group Theory

Group theory is the theory in which you deal with abstract algebraic structures.

Closure property is one of the fundamental properties which defines a group. When you perform any operation on elements within that group, the resultant always comes in the same group.

### Closure Property of Addition

In group theory, the Closure Property is the most basic property of the set. When you do addition within a set, if the sum of any two elements also belongs to that set, it shows closure property under addition. ****For example****,

Consider a number set N (Natural Numbers) under addition

- 2 + 4 = 6 ϵ N
- 4 + 6 = 8 ϵ N

Thus, closure property is followed under Addition operation.

****Learn,**** [****Natural Numbers****](https://www.geeksforgeeks.org/maths/what-are-natural-numbers/)

****Read More,****

> - [Properties of Rational Numbers](https://www.geeksforgeeks.org/maths/properties-of-rational-numbers/)
> - [Properties of Real Numbers](https://www.geeksforgeeks.org/maths/properties-of-real-numbers/)
> - [Properties of Integers](https://www.geeksforgeeks.org/maths/properties-of-integers/)

## Closure Property Examples

****Example 1: Consider the set {1, 2, 3, 4, 5}. Is this set closed under addition?****

****Solution:****

> Yes, it's closed. For instance, 2 + 3 = 5, which is still within the set.

****Example 2: Given a set {6, 12, 18}. Is it closed under multiplication?****

****Solution:****

> Yes, it's closed. Because, 6 * 18 = 108, which is also in the set.

****Example 3: Is the set {1, 3, 5} forms a group under addition (mod 6).****

****Solution:****

> 1 + 3 = 4 (in the set)
> 
> 3 + 5 = 2 (in the set)
> 
> 1 + 5 = 0 (in the set)
> 
> Hence, the set {1, 3, 5} under addition (mod 6) forms a group due to closure.

****Example 4: Compute if {0, 4, 8} is closed under addition (mod 10).****

****Solution:****

> 0 + 4 = 4 (in the set)
> 
> 4 + 8 = 2 (violates closure as 2 is not in the set)
> 
> Hence, {0, 4, 8} is not closed under addition (mod 10).

****Example 5: Verify the closure property of {1, 4, 7} under addition (mod 5).****

****Solution:****

> 1 + 4 = 0 (in the set)
> 
> 4 + 7 = 1 (in the set)
> 
> 1 + 7 = 3 (violates closure as 3 is not in the set)
> 
> Therefore, {1, 4, 7} is not closed under addition (mod 5).

## Closure Property - Practice Questions

****Q1: Find if the set {2, 4, 6, 8} is closed under addition (mod 7).****

****Q2: Show if the set {3, 9, 15, 21} forms a group under multiplication (mod 10).****

****Q3: Verify closure in the set {1, 5, 9, 13} under addition (mod 12).****

****Q4: Determine if the set {0, 5, 10, 15} is closed under multiplication (mod 20).****

****Q5: Find if closure in the set {4, 9, 14, 19} under addition (mod 25).****



- [Closure Property](https://www.geeksforgeeks.org/maths/closure-property/) in Binary Operations
- Associativity of Binary Operations
- Commutativity of Binary Operations
- [Identity Element](https://www.geeksforgeeks.org/maths/identity-property/) of Binary Operations
- Inverse Element of Binary Operations

### Closure Property in Binary Operations

The closure property in binary operation * on set X with element x and y is defined as:

> ****x ∈ X, y ∈ X ⇒ x * y ∈ X****

If x and y belong to a set X then the result of the binary operation between them will also belong to the set X

### Associativity of Binary Operations

Associativity of binary operation * on set X with element x, y and z is defined as:

> ****(x * y) * z = x* (y * z)****

### Commutativity of Binary Operations

Commutativity of binary operation * on set X with element x and y is defined as:

> ****x * y = y * x****

### Identity Element of Binary Operations

Identity element of binary operation * on set X with element x and e is defined as:

> ****x * e = e * x = x****

Then, e is called the identity element.

### Inverse Element of Binary Operations

Inverse element of binary operation * on set X with element x, y and e is defined as:

> ****x * y = y * x = e****

Then x is inverse of y and y is inverse of x.

****Read More,****

- [Commutative Property in Maths](https://www.geeksforgeeks.org/maths/commutative-property/)
- [What is Associative Property](https://www.geeksforgeeks.org/maths/associative-property/)

## Types of Binary Operations

Binary operations are operations which require two inputs. Some of the common types of binary operations are as follows:

- Binary Addition
- [Binary Subtraction](https://www.geeksforgeeks.org/maths/binary-subtraction/)
- [Binary Multiplication](https://www.geeksforgeeks.org/maths/binary-multiplication/)
- Binary Division

Let's discuss these common types in detail as follows:

### Binary Addition

Binary addition is an operation on a set A with elements x and y defined as:

> ****+ : A × A → A such that (x, y) → x + y****

### Binary Subtraction

Consider a set A with elements x and y. Binary subtraction is a closed binary operation on A such that:

> ****- : A × A → A such that (x, y) → x - y****

### Binary Multiplication

Binary multiplication is a binary operation defined on a set A, where each element x and y in A is paired with the operation symbol ×, resulting in x ****×**** y, which belongs to the set A. Mathematically this can be written as:

> ****× : A × A → A such that (x, y) → x × y****

### Binary Division

Binary division on a set A with elements x and y is a binary operation denoted as / and defined as:

> ****/ : A × A → A such that (x, y) → x / y****

## ****Binary Operation Table****

A binary operation table, also known as a Cayley table or operation table, is a systematic way to display the results of applying a binary operation to elements of a set.

In this table, each row represents one of the elements of the set, and each column represents another element. The cell at the intersection of a row and a column contains the result of applying the binary operation to the corresponding pair of elements.

For example, let's consider a set A={0, 1, 2, 3} with addition modulo(⊕) as operation.

|⊕|1|2|3|4|
|---|---|---|---|---|
|1|2|3|4|1|
|2|3|4|1|2|
|3|4|1|2|3|
|4|1|2|3|4|

## Applications of Binary Operations

Some of the common applications of binary operations are:

- Binary operations are fundamental in abstract algebra, where they are used to define algebraic structures such as groups, rings, and fields.
- In combinatorics, binary operations are used to study various counting problems, permutations, and combinations.
- Binary operations are extensively used in computer science for bitwise operations, such as AND, OR, XOR, and complement operations, which are fundamental in digital logic and computer arithmetic.
- In electrical engineering, binary operations are essential for digital signal processing, coding theory, and error detection/correction techniques.

****Read More,****

> - [Set Theory](https://www.geeksforgeeks.org/maths/set-theory/)
> - [Algebraic Structure](https://www.geeksforgeeks.org/engineering-mathematics/groups-discrete-mathematics/)

## Binary Operation Examples

****Example: Consider a binary operation * on set X = {1, 2, 3, 4, 5} defined by x*y = min (x, y). With the help of below table find:****

****(i) Compute (4 * 5) * 1****

****(ii) Is * commutative?****

****(iii) Compute (2 * 5) * (1 * 3)****

****Table:****

|*********|****1****|****2****|****3****|****4****|****5****|
|---|---|---|---|---|---|
|****1****|****1****|****1****|****1****|****1****|****1****|
|****2****|****1****|****2****|****2****|****2****|****2****|
|****3****|****1****|****2****|****3****|****3****|****3****|
|****4****|****1****|****2****|****3****|****4****|****4****|
|****5****|****1****|****2****|****3****|****4****|****5****|

****Solution:****

> ****(i) (4 * 5) * 1****
> 
> From table
> 
> (4 * 5) = 4
> 
> (4 * 5) * 1 = 4 * 1
> 
> (4 * 5) * 1 = 1
> 
> ****(ii) Is * commutative****
> 
> For commutative we have to prove x*y = y*x
> 
> let x = 5 and y = 2
> 
> x*y = 5 * 2 = 2
> 
> y*x = 2 * 5 = 2
> 
> Therefore, * is commutative.
> 
> ****(iii) (2 * 5) * (1 * 3)****
> 
> From table
> 
> (2 * 5) = 2
> 
> (1 * 3) = 1
> 
> (2 * 5) * (1 * 3) = 2 * 1
> 
> ****(2 * 5) * (1 * 3) = 1****

## Practice Problems on Binary Operations

****Problem : Consider a binary operation * on set X = {a, b, c} defined by below. Find:****

****(i) Compute (a * b) * c****

****(ii) Is * commutative?****

****(iii) Find the identity element of the binary operation.****

****Table****:

|*|a|b|c|
|---|---|---|---|
|a|a|b|c|
|b|b|c|a|
|c|c|a|b|

## Conclusion

Binary operations are essential math ideas used in many areas like algebra, computer science, engineering, and cryptography. Understanding how these operations work and their properties is essential for solving complex problems and building efficient algorithms. This article gave a basic introduction to this important concept including topics such as properties and types of binary operations.


- Binary subtraction is a fundamental idea in [binary operations](https://www.geeksforgeeks.org/maths/binary-operation/).
- There are two components in binary subtraction: ****0**** and ****1****.

### ****Is it Possible to Subtract Binary Numbers?****

Yes, binary number subtraction is feasible. It's pretty similar to subtracting base ten values. When you combine 1 + 1 + 1, you get 3. However, in a [binary number system](https://www.geeksforgeeks.org/maths/binary-number-system/), the sum of 1 + 1 + 1 equals 1 1. We must be careful while subtracting or adding in this case since it might get complicated. Base-2 is used to express a binary number. The basic subtraction table of binary numbers is,

|x|y|x - y|
|---|---|---|
|0|0|0|
|0|1|1 (Borrow 1)|
|1|0|1|
|1|1|0|

### Binary Subtraction Definition

Binary numbers are a base-2 numeral system that represents data with only two symbols, commonly 0 and 1. the subtraction of two binary digits i called as Binary Subtraction and binary subtraction follow some fundamental rules that are added below.

Before moving any further we must in brief learn about Binary Numbers.

### What are Binary Numbers?

Binary Numbers are numbers that uses two symbols "0" and "1" to write all the numbers. This system is used by computer programmers to write various computer code.

Binary numbers are expressed in the base-2 numeral system. Each digit in the ****high-order**** Number System is called a ****bit****.

> ****Learn more about -**** [****Binary Numbers****](https://www.geeksforgeeks.org/maths/binary-number-system/)

## Binary Subtraction Rules

Binary Subtraction is performed in the same manner as decimal subtraction. However, there are some specific rules regarding the subtraction among the binary digits 0 and 1 which we need to follow while performing Binary Subtraction.

### Binary Subtraction Rules Table

Binary subtraction is easily achieved using the rules added in the table below,

|Table of Binary Subtraction Rule|   |   |
|---|---|---|
|****Binary Number****|****Subtraction Value****|Rule|
|---|---|---|
|0 - 0|0|When we subtract 0 from 0, we get 0|
|1 - 0|1|When we subtract 0 from 1, we get 1|
|0 - 1|1 (Take 1 from the following high-order digit)|When we subtract 1 from 0, we get 1 with a borrow of 1|
|1 - 1|0|When we subtract 1 from 1, we get 0|

The addition of two binary numbers, 1 and 1, results in 10, where 1 is taken to the next high order and 0 is disregarded. However, nothing is carried over as the result of subtracting 1 from 1 is 0. When subtracting 1 from 0 in decimal subtraction, we borrow 1 from the preceding number to make it 10, and the outcome is 9 as 10 - 1 = 9. Nevertheless, binary subtraction yields just one result.

## Methods of Binary Subtraction

Binary numbers can be represented as decimal or base-10 numbers. Computers use binary numerals to represent data because they can only comprehend the binary digits 0 and 1. With the example below, let's learn how to subtract binary numbers.

### Method 1: Subtraction of Binary Numbers without Borrowing

In binary, the number 8 is represented as (1000)2 and the number 25 is represented as (11001)2. Now subtract (1000)2 from (11001)2.

****Step 1:**** Put the numbers in the order given below.

![Binary-Subtraction-Method 1 Step-1](https://media.geeksforgeeks.org/wp-content/uploads/20240123161107/Binary-Subtraction-Step-1.jpg)

****Step 2:**** To subtract the numbers, use binary subtraction principles.

- Let us begin this subtraction by subtracting the integers on the right and working our way up to the next ****higher-order**** digit.
- To begin, subtract (1-0). This equals 1.
- Likewise, we proceed to the next higher-order digit and subtract (0-0), which is ****equal**** to 0.
- Again, subtract (0-0), which is equals to 0. Then, we subtract (1-1), the outcome is 0.
- As a result, the difference is 100012.

![Binary-Subtraction-Method 2 Step-2](https://media.geeksforgeeks.org/wp-content/uploads/20240123161301/Binary-Subtraction-Step-2.jpg)

(10001)2 has a decimal equivalent of 17. As a result, the distinction is accurate.

### Method 2: Subtraction of Binary Numbers With ****b****orrowing

In binary, the number 12 is represented as (1100)2 and the number 26 is represented as (11010)2. Now subtract (1100)2 from (11010)2.

****Step 1:**** Put the numbers in the order given below.

![Binary-Subtraction Method 2 Step-1](https://media.geeksforgeeks.org/wp-content/uploads/20240123161423/Binary-Sub-Step-1.jpg)

****Step 2:**** To subtract the numbers, use binary subtraction principles.

- Start the subtraction from the rightmost digit and move toward the higher-order digits(leftwards).
- Subtract (0 - 0) → Result: ****0****
- Subtract (1 - 0) → Result: ****1****
- Subtract (0 - 1):  
    1. Borrow 1 from the next higher order digit. The higher order digit will become 0.  
    2. The borrowed place will now have 10. ( equivalent to 2 in decimal)  
    3. Now do the operation 10 - 1 = 1  
    Thus, Result: ****1****
- Subract (0 (borrowed earlier) - 1):  
    1. Borrow 1 from the next higher order digit. The higher order digit will become 0.  
    2. The borrowed place will now have 10. ( equivalent to 2 in decimal)  
    3. Now do the operation 10 - 1 = 1  
    Thus, Result: ****1****
- Next digit is 0(borrowed in previous step). → Result: ****0****
- Final binary difference: ****(1110)₂****

![Binary-Subtraction Method 2-Step-2](https://media.geeksforgeeks.org/wp-content/uploads/20240123161604/Binary-Sub-Step-2.jpg)(1110)2 has a decimal equivalent of 14.

## Binary Subtraction Using 1's Complement

A number's 1's complement is derived by reversing every 0 to 1 and every 1 to 0 in a binary integer. For example, the binary number 1102 has a 1's complement of 0012. Please follow the instructions below to accomplish binary subtraction using 1's complement. Binary subtraction with 1's complement entails adding the complement of the subtrahend (the number being subtracted) to the minuend (the number being subtracted from). Here's an illustration:

Let's subtract (100010)2​ from (110110)2​. In this case, the binary equivalent of 34 is (100010)2 whereas the binary equivalent of 54 is (110110)2.

****Step 1:**** First identify the Minuend and Subtrahend. In this,****,**** Minuend is (110110)2 and subtrahend is (100010)2.

****Step 2:**** Find the 1's complement of the subtrahend. The subtrahend is (100010)2 and after 1's complement it is (011101)2.

****Step 3:**** Now add the minuend and the 1's complement of the subtrahend.

![Binary-Subtraction-1's-Complement-Step-1](https://media.geeksforgeeks.org/wp-content/uploads/20240123162708/Binary-Subtraction-1's-Complement-Step-1.jpg)

****Step 4:**** This increment is shown in the left-most digit, 1. We add that to the result, which is (010011)2.

![Binary-Subtraction-1's-Complement-Step-2](https://media.geeksforgeeks.org/wp-content/uploads/20240123162830/Binary-Subtraction-1's-Complement-Step-2.jpg)

As a consequence, the answer is (10100)2. In addition, the difference between 54 and 34 is 20. The binary equivalent of 20 is (10100)2.

## Binary Subtraction Using 2's Complement

Binary subtraction with 2's complement entails adding the complement of the subtrahend (the number being subtracted) to the minuend (the amount being removed). Here's an illustration:

Let's subtract (100010)2​ from (110110)2​. In this case, the binary equivalent of 34 is (100010)2 whereas the binary equivalent of 54 is (110110)2.

****Step 1:**** First identify the Minuend and Subtrahend. In ****thi****s Minuend is (110110)2 and subtrahend is (100010)2.

****Step 2:**** Find the 1's complement of the subtrahend. The subtrahend is (100010)2 and after 1's complement it is (011101)2. Now add 1 to the 1's complement (011101)2 + 1 = (011110)2. So the 2's Complement is (011110)2.

****Step 3:**** Now add the minuend and the 2's complement of the subtrahend.

![Binary-Subtraction-2's-Complement](https://media.geeksforgeeks.org/wp-content/uploads/20240123163002/Binary-Subtraction-2's-Complement.jpg)

In this leave the carryout. As a consequence, the answer is (10100)2. In addition, the difference between 54 and 34 is 20. The binary equivalent of 20 is (10100)2

****Read More,****

- [****Binary Division****](https://www.geeksforgeeks.org/maths/binary-division/)
- [****Binary Multiplication****](https://www.geeksforgeeks.org/maths/binary-multiplication/)

## Solved Examples ****of**** Binary Subtraction

Various examples ****of**** binary subtraction are given below:

****Example 1:**** Subtract (1011010)2 - (001010)2

****Solution:****

> 1011010  
> -001010  
> --------  
> 1010000  
> --------
> 
> After subtracting (1011010)2 - (001010)2 = (1010000)2

****Example 2:**** Subtract (1110)₂ - (11)₂ using 1’s complement.

****Solution:****

> ****Step 1:**** First find the subtrahend and Minuend. Subtrahend is 11 and Minuend is 1110.
> 
> ****Step 2:**** Now calculate the 1’s complement of the subtrahend and then add that with minuend.
> 
> 1110  
> +1100  
> ------  
> 11010  
> ------
> 
> ****Step 3:**** Here, carry over occurs, move it to the least significant portion.
> 
> 1010  
> +1  
> ----  
> 1011  
> ----
> 
> Hence the answer is (1011)2.

****Example 3:**** Subtract 101₂ - 110₂ using 2’s complement.

****Solution:****

> ****Step 1:**** First find the subtrahend and Minuend. Subtrahend is (101)2 and Minuend is (110)2.
> 
> ****Step 2:**** Find the 1's complement of the subtrahend. The subtrahend is (101)2 and after 1's complement it is (010)2. Now add 1 to the 1's complement (010)2 + 1 = (011)2. So the 2's Complement is (011)2.
> 
> ****Step 3:**** Now add the minuend and the 2's complement of the subtrahend.
> 
> 110  
> +011  
> ----
> 
> 1001  
> ----
> 
> In this leave the carry. Hence the answer is (001)2.

## Practice Problems on Binary Subtraction

Some practice problems on Binary Subtraction are

****Problem 1:**** Do Binary Subtraction: (101100)2 - (100010)2.

****Problem 2:**** Subtract (10110)₂ - (1110)₂ using 1’s complement.

****Problem 3:**** Subtract (100100)2 - (10010)3.

****Problem 4:**** Subtract (11001)₂ - (1000)₂ using 2’s complement.


- [Binary Subtraction](https://www.geeksforgeeks.org/maths/binary-subtraction/)
- [Binary Multiplication](https://www.geeksforgeeks.org/maths/binary-multiplication/)
- [Binary Division](https://www.geeksforgeeks.org/maths/binary-division/)

Now, let's learn about the same in detail.

### ****Binary Addition****

The result of the addition of two binary numbers is also a binary number. To obtain the result of the addition of two binary numbers, we have to add the digits of the binary numbers digit by digit. The table below shows the rules of binary addition.

|Binary Number (1)|Binary Number (2)|Addition|Carry|
|---|---|---|---|
|0|0|0|0|
|0|1|1|0|
|1|0|1|0|
|1|1|0|1|

****Example:**** Find (1101)2 + (1011)2 = ?

> 1101(13 in decimal)+1011(11 in decimal)11000(24 in decimal)+​1​111​100​010​1(13 in decimal)1(11 in decimal)0(24 in decimal)​​

### ****Binary Subtraction****

The result of the subtraction of two binary numbers is also a binary number. To obtain the result of the subtraction of two binary numbers, we have to subtract the digits of the binary numbers digit by digit. The table below shows the rule of binary subtraction.

|Binary Number (1)|Binary Number (2)|Subtraction|Borrow|
|---|---|---|---|
|0|0|0|0|
|0|1|1|1|
|1|0|1|0|
|1|1|0|0|

****Example:**** Find (1011)2 - (1110)2 = ?

> 1110(14 in decimal)−1011(11 in decimal)0011(3 in decimal)−​110​100​111​0(14 in decimal)1(11 in decimal)1(3 in decimal)​​

### ****Binary Multiplication****

The multiplication process of binary numbers is similar to the multiplication of decimal numbers. The rules for multiplying any two binary numbers are given in the table.

|Binary Number (1)|Binary Number (2)|Multiplication|
|---|---|---|
|0|0|0|
|0|1|0|
|1|0|0|
|1|1|1|

****Example:**** Find (101)2 ⨉ (11)2 = ?

****Solution:****

> 101×11101+101×1111×+​11​1101​01011​111×1​​

### ****Binary Division****

The [division method](https://www.geeksforgeeks.org/maths/long-division/) for binary numbers is similar to that of the decimal number division method. Let us go through an example to understand the concept better.

****Example:**** Find (11011)2 ÷ (11)2 = ?

> ![Divide binary numbers (101101) by (110)](https://media.geeksforgeeks.org/wp-content/uploads/20220718160746/bs1-221x300.jpg)

## 1's and 2's Complement of a Binary Number

- [****1's Complement****](https://www.geeksforgeeks.org/digital-logic/ones-complement/) ****of a Binary Number is obtained by inverting the digits of the binary number.****

****Example: Find the 1's complement of (10011)********2********.****

****Solution:****

> Given Binary Number is (10011)2
> 
> Now, to find its 1's complement, we have to invert the digits of the given number.
> 
> To find the ****1's complement**** of a binary number, you simply ****flip all the bits****:
> 
> Thus, 1's complement of (10011)2 is (01100)2

- [****2's Complement****](https://www.geeksforgeeks.org/digital-logic/twos-complement/) ****of a Binary Number is obtained by inverting the digits of the binary number and then**** ****adding 1 to the least significant bit.****

****Example: Find the 2's complement of (1011)********2********.****

****Solution:****

> Given Binary Number is (1011)2
> 
> To find the 2's complement, first find its 1's complement, i.e., (0100)2
> 
> Now, by adding 1 to the least significant bit, we get (0101)2
> 
> Hence, the 2's complement of (1011)2 is (0101)2

## Uses of the Binary Number System

Binary Number Systems are used for various purposes, and the most important use of the binary [number system](https://www.geeksforgeeks.org/maths/number-system-in-maths/) is,

- Binary Number System is used in all Digital Electronics for performing various operations.
- Programming Languages use the Binary Number System for encoding and decoding data.
- Binary Number System is used in Data Sciences for various purposes, etc.

### ****Also Check:****

> - [Binary Formula](https://www.geeksforgeeks.org/maths/binary-formula/)
> - [Decimal vs Binary Number System](https://www.geeksforgeeks.org/maths/decimal-vs-binary/)

## Solved Example of the Binary Number System

****Example 1: Convert the Decimal Number (98)********10**** ****into Binary.****  
****Solution:**** 

> ![Convert Decimal Number (98) into Binary](https://media.geeksforgeeks.org/wp-content/uploads/20220718154330/bs3-258x300.jpg)
> 
> Thus, Binary Number for (98)10 is equal to (1100010)2

****Example 2:**** ****Convert**** ****the Binary Number (1010101)********2**** ****to a decimal**** ****Number.****  
****Solution:**** 

> Given Binary Number, (1010101)2
> 
> = (1 × 20) + (0 × 21) + (1 × 22) + (0 × 23) + (1 × 24) + (0 × 25) + (1 ×26)  
> = 1 + 0 + 4 + 0 + 16 + 0 + 64  
> = (85)10
> 
> Thus, Binary Number (1010101)2 is equal to (85)10 in decimal system.

****Example 3: Divide (11110)********2**** ****by (101)********2********.****  
****Solution:****

> ![Divide binary numbers (11110) by (101)](https://media.geeksforgeeks.org/wp-content/uploads/20220718140851/bs-300x247.jpg)

****Example 4: Add (11011)********2**** ****and (10100)********2********.****  
****Solution:****

> ![Add binary numbers (11011) and (10100)](https://media.geeksforgeeks.org/wp-content/uploads/20220718141658/bs1.jpg)
> 
> Hence, (11011)2 + (10100)2 =  (101111)2

****Example 5: Subtract (11010)********2**** ****and (10110)********2********.****  
****Solution:**** 

> ![Subtract binar numbers (11010) and (10110)](https://media.geeksforgeeks.org/wp-content/uploads/20220718142253/bs11.jpg)
> 
> Hence, (11010)2 - (10110)2 = (00100)2

****Example 6: Multiply (1110)********2**** ****and (1001)********2********.****  
****Solution:**** 

> ![Multiply binary numbers (1110) and (1001)](https://media.geeksforgeeks.org/wp-content/uploads/20220718143648/bs12-136x200.jpg)
> 
> Thus, (1110)2 × (1001)2 = (1111110)2

****Example 7: Convert (28)********10**** ****into a binary number.****

****Solution:****

> ![Convert (28) into a binary number.](https://media.geeksforgeeks.org/wp-content/uploads/20220718153639/bs2-291x300.jpg)
> 
> Hence, (28)10 is expressed as (11100)2.

****Example 8: Convert (10011)********2**** ****to a decimal number.****  
****Solution:****

> The given binary number is (10011)2.
> 
> (10011)2 = (1 × 24) + (0 × 23) + (0 × 22) + (1 × 21) + (1 × 20)
> 
> = 16 + 0 + 0 + 2 + 1 = (19)10
> 
> Hence, the binary number (10011)2 is expressed as (19)10.

## Practice Pr****o****blem Based on Binary Number

****Question 1.**** Convert the decimal number ****(98)₁₀**** into binary.  
****Question 2.**** Divide the binary number ****(11110)₂**** by ****(101)₂****  
****Question 3.**** Find the 2's complement of the binary number ****(1011)₂****.  
****Question 4.**** Multiply the binary numbers ****(1110)₂**** and ****(1001)₂****.  
****Question 5.**** Subtract the binary numbers ****(11010)₂**** and ****(10110)₂****.  
****Question 6.**** Add the binary numbers ****(11011)₂**** and ****(10100)₂****.

****Answer:-****

> ****1. (1100010)********2********​, 2. (110)********2********​, 3. (0101)********2********, 4. (1111110)********2********, 5. (00100)********2********​​, 6. (101111)********2****

Suggested Quiz

4 Questions

What is the decimal equivalent of the binary number 101101?

- A
    
    45
    
- B
    
    43
    
- C
    
    46
    
- D
    
    44
    

Which of the following is a correct binary addition?

- A
    
    1011 + 1101 = 11000
    
- B
    
    1010 + 1000 = 11010
    
- C
    
    1001 + 0011 = 11001
    
- D
    
    1011 + 1101 = 10010
    

What is the result of binary multiplication 101 × 11?

- A
    
    1111
    
- B
    
    1110
    
- C
    
    10011
    
- D
    
    1001
    

Simplify (1011 ⋅ 100) + (101 ⋅ 110).

- A
    
    1101010
    
- B
    
    1010101
    
- C
    
    1001010
    
- D
    
    0101010
    

![](https://media.geeksforgeeks.org/auth-dashboard-uploads/sucess-img.png)

Quiz Completed Successfully

Your Score :  2/4

Accuracy : 0%

View Explanation

1/4< Previous Next >

Binary System is a system of writing numbers using only two numbers that are, 0 and 1. The base of the binary number is 2. This system was first used by ancient Indian, Chinese, and Egyptian people for various purposes. The [binary number system](https://www.geeksforgeeks.org/maths/binary-number-system/) is used in electronic and computer programming.

### What is Decimal System?

The Decimal Numbers System is the number system that is used by us in our daily lives. The base of Decimal numbers is 10 and it uses 10 digits that are 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9.

## How to use Binary to Decimal Calculator?

We can easily use the ****binary-to-decimal calculator**** by following the steps discussed below,

****Step 1:**** Enter the given value in the binary input field.

****Step 2:**** Click on the convert button to convert the binary value into the decimal value.

****Step 3:**** The value shown as the result is the required value in the decimal form.

## Binary to Decimal Formula

To convert a ****binary number to decimal**** we need to perform a multiplication operation on each digit of a binary number from right to left with powers of 2 starting from 0 and add each result to get the ****decimal**** number of it. 

****Decimal Number = n********th**** ****bit × 2********n-1****

### Binary to Decimal Formula

> ****n = b********n********q + b********n-1********q********n-2**** ****+.........+ b********2********q********2**** ****+b********1********q********1**** ****+b********0********q********0**** ****+ b********-1********q********-1**** ****+ b********-2********q********-2**** 

Where,

- N is Decimal Equivalent
- b is the Digit
- q is the Base Value

## How to Convert Binary to Decimal

You just have to follow the below steps to convert binary numbers to their decimal equivalent.

****Step 1:**** Write the binary number and count the powers of 2 from right to left (starting from 0).

****Step 2****: Write each binary digit(right to left) with corresponding powers of 2 from right to left, such that MSB or the first binary digit will be multiplied by the greatest power of 2.

****Step 3****: Add all the products in the step 2

****Step 4****: The answer is our decimal number.

This can be better explained using the below examples.

## Binary to Decimal Conversion

Binary to Decimal conversion is achieved using the two steps that are,

- Positional Notation Method
- Doubling Method

Now let's learn about them in detail.

## Method 1: Using Positions

Binary to Decimal Conversion can be achieved using the example added below.

****Example 1:**** ****Let's consider a binary number 1111. We need to convert this binary number to a decimal number.****

> As mentioned in the above paragraph while converting from binary to decimal we need to consider each digit in binary number from right to left.
> 
> ![Convert 1111 to Decimal](https://media.geeksforgeeks.org/wp-content/uploads/20211220191513/binary.PNG)
> 
> By this way, we can do binary to decimal conversion.

****Note:**** We represent any binary number with this format (xxxx)2 and decimal in (xxxx)10 format.

****Example 2: Convert (101010)********2**** ****= (?)********10****

> ![Convert 101010 to decimal](https://media.geeksforgeeks.org/wp-content/uploads/20211220193034/binary2.PNG)
> 
> We keep on increasing the power of 2 as long as number of digits in binary number increases.

****Example 3: Convert (11100)********2**** ****= (?)********10****

> ![Convert 11100 to Decimal](https://media.geeksforgeeks.org/wp-content/uploads/20211220194742/binary3.PNG)
> 
> Resultant Decimal number = 0+0+4+8+16 = 28
> 
> So (11100)2 = (28)10

****Also Check,****

- [Decimal to Binary Converter](https://www.geeksforgeeks.org/utilities/decimal-to-binary/)

## Method 2: Doubling Method

To explain this method we will consider an ****example**** and try to solve that stepwise.

****Example 1: Convert Binary number (10001)********2**** ****to decimal.****

> Similar to the above approach, In this approach also consider each digit but from left to right and performs step-wise computations on it.
> 
> |   |   |   |   |   |
> |---|---|---|---|---|
> |****1****|0|0|0|1|
> 
> ****Step-1**** First we need to multiply 0 with 2 and add the 1st digit in binary number.
> 
> 0 x 2 + ****1**** = 0 + 1 = 1
> 
> ****Step-2**** Now use the result of above step and multiply with 2 and add the second digit of binary number.
> 
> |   |   |   |   |   |
> |---|---|---|---|---|
> |1|****0****|0|0|1|
> 
> 1 x 2 + ****0 =**** 2 + 0 = 2
> 
> The same step 2 is repeated until there will be no digit left. The final result will be the resultant decimal number.
> 
> |   |   |   |   |   |
> |---|---|---|---|---|
> |1|0|****0****|0|1|
> 
> 2 x 2 + ****0**** = 4 + 0 = 4
> 
> |   |   |   |   |   |
> |---|---|---|---|---|
> |1|0|0|****0****|1|
> 
> 4 x 2 + ****0**** = 8 + 0 = 8
> 
> |   |   |   |   |   |
> |---|---|---|---|---|
> |1|0|0|0|****1****|
> 
> 8 x 2 + ****1**** = 16 + 1 = 17
> 
> So we performed step 2 on all remaining numbers and finally, we left with ****result 17 which is a decimal number for the given binary number.****
> 
> So ****(10001)********2**** ****= (17)********10****

****Example 2: Convert (111)********2**** ****to decimal using doubling approach.****

> |   |   |   |
> |---|---|---|
> |****1****|1|1|
> 
> 0 x 2 + ****1**** = 0 + 1 = 1
> 
> |   |   |   |
> |---|---|---|
> |1|****1****|1|
> 
> 1 x 2 + ****1**** = 2 + 1 = 3
> 
> |   |   |   |
> |---|---|---|
> |1|1|****1****|
> 
> 3 x 2 + ****1**** = 6 + 1 = 7
> 
> The final result is 7 which is a Decimal number for 111 ****binary numeral system****. So ****(111)********2**** ****= (7)********10****
> 
> These are the 2 approaches that can be used or applied to convert binary to decimal.

## How to Read a Binary Number?

Binary numbers are read by separating them into separate digits. Each digit in binary is represented using 0 and 1 and they are the powers of 2 starting from left hand side and then the power is gradually increased from 0 to (n-1).

## ****Binary to Decimal Conversion Table****

The given ****binary to decimal conversion table**** will help you to ****convert binary to decimal****.

|Decimal Number|Binary Number|
|---|---|
|0|0|
|1|1|
|2|10|
|3|11|
|4|100|
|5|101|
|6|110|
|7|111|
|8|1000|
|9|1001|
|10|1010|
|11|1011|
|12|1100|
|13|1101|
|14|1110|
|15|1111|
|16|10000|
|17|10001|
|18|10010|
|19|10011|
|20|10100|
|21|10101|
|22|10110|
|23|10111|
|24|11000|
|25|11001|
|26|11010|
|27|11011|
|28|11100|
|29|11101|
|30|11110|
|31|11111|
|32|100000|
|64|1000000|
|128|10000000|
|256|100000000|

## Conclusion

In conclusion, the ****Binary to Decimal Calculator**** is a free online tool prepared by GeekforGeeks that converts the given value of the ****binary number system**** into the value of a ****decimal number system**** . It is a fast and easy-to-use tool that helps students solve various problems.

****Read More,****

- [Decimal to Hexadecimal Converter](https://www.geeksforgeeks.org/utilities/decimal-to-hex-converter/)
- [Binary to Hexadecimal Converter](https://www.geeksforgeeks.org/maths/how-to-convert-binary-to-hexadecimal/)

## Binary to Decimal Conversion Examples

### ****Example 1: Convert (111)********2**** ****to Decimal.****

### ****Solution:****

> We have (111)2 in binary
> 
> ⇒ 1 ⨯ 22 + 1 ⨯ 21 + 1 ⨯ 20
> 
> = 4 + 2 + 1 = 7

### ****Example 2: Convert (10110)********2**** ****to Decimal.****

### ****Solution:****

> We have (10110)2 in Binary
> 
> 1 ⨯ 24 + 0 ⨯ 23 + 1 ⨯ 22 + 1 ⨯ 21 + 0 ⨯ 20
> 
> = 16 + 4 + 2 = 22

### ****Example 3: Convert (10001)********2**** ****to Decimal.****

### ****Solution:****

> We have (10001)2 in Binary
> 
> ⇒ 1 ⨯ 24 + 0 ⨯ 23 + 0 ⨯ 22 + 0 ⨯ 21 + 1 ⨯ 20
> 
> = 16 + 0 + 0 + 0 + 1 = 17

### ****Example 4: Convert (1010)********2**** ****to Decimal.****

### ****Solution:****

> We have (1010)2 in Binary
> 
> ⇒ 1 ⨯ 23 + 0 ⨯ 22 + 1 ⨯21 + 0 ⨯ 20
> 
> = 0 + 8 + 2 + 0 = 10

### ****Example 5: Convert (10101101)********2**** ****to Decimal.****

### ****Solution:****

![binary-to-decimal](https://media.geeksforgeeks.org/wp-content/uploads/20231101152126/binary-to-decimal.webp)

## Convert Binary to Decimal (bn to dec)

### ****Q1: Convert (11000)********2**** ****to Decimal.****

### ****Q2: Convert (10111)********2**** ****to Decimal.****

### ****Q3: Convert (111110000)********2**** ****to Decimal.****

### ****Q4: Convert (00011)********2**** ****to Decimal.****

### ****Q5: Convert (110011)********2**** ****to Decimal.****

****Also Check,****

- [****Million to Crore****](https://www.geeksforgeeks.org/utilities/million-to-crore-converter/)
- [****Million to Lakhs****](https://www.geeksforgeeks.org/utilities/million-to-lakhs-converter/)


> Also, Check other Converters
> 
> - [Binary to Decimal](https://www.geeksforgeeks.org/utilities/binary-to-decimal/)
> - [Decimal to Hexadecimal](https://www.geeksforgeeks.org/utilities/decimal-to-hex-converter/)

## Decimal to Binary Conversion

Before learning how to ****convert decimal to binary**** in a number system, let's first understand what a [decimal number system](https://www.geeksforgeeks.org/digital-logic/decimal-number-system/) is and what is a [binary number system](https://www.geeksforgeeks.org/maths/binary-number-system/).

### ****Decimal Number System****

> The number system that has a base value of 10 is called Decimal Number System. Decimal Numbers are consist of the following digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.

### ****Binary Number System****

> A binary Number System is a base-2 number system that uses two states 0 and 1 to represent a number. For example: 01, 111, etc.

All the decimal numbers have their corresponding binary numbers. These binary numbers are used in computer applications and for programming or coding purposes. This is because binary digits, 0 and 1 are only understood by computers.

### ****How to Convert Decimal to Binary****

To convert decimal numbers into binary, there are several methods, including formulas and division techniques. In this explanation, we'll use the remainder method. The steps to convert a decimal number to binary using this method are as follows:

****Step 1:**** Divide the given decimal number by 2, and find the remainder (Ri).

****Step 2:**** Now divide the quotient (Qi) that is obtained in the above step by 2, and find the remainder.

****Step 3:**** Repeat the above steps 1 and 2 until 0 is obtained as a quotient.

****Step 4:**** Write down the remainder in the following manner: the last remainder is written first, followed by the rest in reverse order (Rn, R(n - 1) .... R1). Thus binary conversion of the given decimal number will be obtained.

Let's understand the above steps with the help of an example.

****Example: Convert 17 to Binary Form.****

****Solution:****

> Following the above steps we will divide 17 by 2 successively. The division process is shown the image added below:
> 
> ![17-in-Binary](https://media.geeksforgeeks.org/wp-content/uploads/20230905164644/17-in-Binary.png)
> 
> Hence the Binary Equivalent of 17 is 10001.

## Decimal to Binary Conversion Table

The common numbers in the Decimal number system and their corresponding binary number, along with the hexadecimal form, are as follows:

|Decimal Number|Binary Number|Hexadecimal Number|
|---|---|---|
|0|0|0|
|1|1|1|
|2|10|2|
|3|11|3|
|4|100|4|
|5|101|5|
|6|110|6|
|7|111|7|
|8|1000|8|
|9|1001|9|
|10|1010|A|
|11|1011|A|
|12|1100|C|
|13|1101|D|
|14|1110|E|
|15|1111|F|
|16|10000|10|
|17|10001|11|
|18|10010|12|
|19|10011|13|
|20|10100|14|
|21|10101|15|
|22|10110|16|
|23|10111|17|
|24|11000|18|
|25|11001|19|
|26|11010|1A|
|27|11011|1B|
|28|11100|1C|
|29|11101|1D|
|30|11110|1E|
|31|11111|1F|
|32|100000|20|
|64|1000000|40|
|128|10000000|80|
|256|100000000|100|

## Decimal to Binary Solved Examples

Some examples of converting decimal numbers to binary are:

### ****Decimal 10 to Binary****

> ****Divide 10 by 2****:  
> Quotient = ****5****, Remainder = ****0****.
> 
> 1. ****Divide 5 by 2****:  
>     Quotient = ****2****, Remainder = ****1****.
> 2. ****Divide 2 by 2****:  
>     Quotient = ****1****, Remainder = ****0****.
> 3. ****Divide 1 by 2****:  
>     Quotient = ****0****, Remainder = ****1****.
> 
> Now, write the remainders in reverse order:  
> ****Binary of 10**** = ****1010****.

### ****Decimal 25 to Binary****

> ****Divide 25 by 2****:  
> Quotient = ****12****, Remainder = ****1****.
> 
> 1. ****Divide 12 by 2****:  
>     Quotient = ****6****, Remainder = ****0****.
> 2. ****Divide 6 by 2****:  
>     Quotient = ****3****, Remainder = ****0****.
> 3. ****Divide 3 by 2****:  
>     Quotient = ****1****, Remainder = ****1****.
> 4. ****Divide 1 by 2****:  
>     Quotient = ****0****, Remainder = ****1****.
> 
> Reading the remainders in reverse order: ****11001****.  
> ****Therefore, the binary equivalent of decimal 25 is 11001.****

### ****Decimal 47 to Binary****

> ****Divide 47 by 2****:  
> Quotient = ****23****, Remainder = ****1****.
> 
> 1. ****Divide 23 by 2****:  
>     Quotient = ****11****, Remainder = ****1****.
> 2. ****Divide 11 by 2****:  
>     Quotient = ****5****, Remainder = ****1****.
> 3. ****Divide 5 by 2****:  
>     Quotient = ****2****, Remainder = ****1****.
> 4. ****Divide 2 by 2****:  
>     Quotient = ****1****, Remainder = ****0****.
> 5. ****Divide 1 by 2****:  
>     Quotient = ****0****, Remainder = ****1****.
> 
> Reading the remainders in reverse order: ****101111****.  
> ****Therefore, the binary equivalent of decimal 47 is 101111.****

## Conclusion

In Conclusion, Decimal to Binary Calculator is a free online tool prepared by GeekforGeeks that converts the given value of the decimal number into the value of the binary number (0,1). It is a fast and easy-to-use tool that helps students solve various problems.

## Solved Questions on Decimal to Binary Conversion

****Question**** (****278)********10**** ****in Binary?****

****Solution:****

> We have 278 in Decimal. To Convert in Binary we will divide 278 successively by 2.
> 
> ![278-in-Binary](https://media.geeksforgeeks.org/wp-content/uploads/20230905164109/278-in-Binary.png)
> 
> Hence, (278) in decimal is equivalent to (100010110) in binary.

****Question**** ****(25)********10**** ****in Binary****

****Solution:****

> We have 25 in decimal. To convert 25 in binary we will divide 25 by 2 successively
> 
> ![25-in-Binary](https://media.geeksforgeeks.org/wp-content/uploads/20230905165632/25-in-Binary.png)
> 
> Hence, the Binary Equivalent of 25 is 11001

****Question (75)********10**** ****to binary?****

****Solution:****

> We have 75 in decimal. To convert 75 in binary we will divide 25 by 2 successively
> 
> ![75-in-Binary](https://media.geeksforgeeks.org/wp-content/uploads/20230905170606/75-in-Binary.png)
> 
> Hence, the Binary Equivalent of 75 is 1001011

## Practice Problems on Decimal to Binary Conversion

****Question 1: Convert 248 in Binary.****

****Question 2: Convert 575 in Binary.****

****Question 3: What is the**** decimal ****equivalent of 49?****

****Question 4: Convert (56)********10**** ****to (....)********2********.****

****Question 5: What is the Binary Form of 95?****

| Related Articles                                                                                 |                                                                                                   |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| [Decimal to Octal](https://www.geeksforgeeks.org/maths/convert-decimal-to-octal/)                | [Decimal to Fraction](https://www.geeksforgeeks.org/maths/convert-decimal-to-fraction/)           |
| [Million to Lakh Converter](https://www.geeksforgeeks.org/utilities/million-to-lakhs-converter/) | [Million to Crore Converter](https://www.geeksforgeeks.org/utilities/million-to-crore-converter/) |
> 1. [Decimal to Binary Conversion](https://www.geeksforgeeks.org/utilities/decimal-to-binary/) and its [implementation](https://www.geeksforgeeks.org/dsa/program-decimal-binary-conversion/).
> 2. [Decimal to Octal Conversion](https://www.geeksforgeeks.org/maths/convert-decimal-to-octal/) and its [implementation](https://www.geeksforgeeks.org/dsa/program-decimal-octal-conversion/).
> 3. [Decimal to Hexadecimal Conversion](https://www.geeksforgeeks.org/utilities/decimal-to-hex-converter/) and its [implementation](https://www.geeksforgeeks.org/dsa/program-decimal-hexadecimal-conversion/).

Beyond conversions, the decimal system also supports techniques like complements, used in subtraction and error detection.

## ****9’s and 10’s Complement of Decimal (Base-10) Number****

Complements are used in number systems to simplify subtraction and error detection in digital systems. In the decimal system, the ****9’s complement**** and ****10’s complement**** are commonly used.

****Steps to Find 9’s Complement:****

1. Write the given decimal number.
2. Subtract each digit from 9.

****Example : 9’s Complement of 2020****

 9 - 2 = 7  
 9 - 0 = 9  
 9 - 2 = 7  
 9 - 0 = 9

Thus, the 9’s complement of 2020 is 7979.

****Steps to Find 10’s Complement:****

1. Find the 9’s complement of the number.
2. Add 1 to the least significant digit (LSB).

****Example : 10’s Complement of 2020****

- 9’s complement of 2020 = 7979
- Adding 1: 7979 + 1 = 7980

Thus, the 10’s complement of 2020 is 7980

## Practice Question on Decimal Number System

****Question 1:**** Convert the binary system 1102 to its decimal equivalent.

****Question 2:**** Find the 10's complement of 725,

****Question 3:**** Convert the decimal number 29 to binary.

****Question 4:**** Find the 9's complement of decimal number 486.

****Question 5:**** Covert 42310 to hexadecimal.

### ****Answer key:****

> 1. 610  
> 2. 275  
> 3. 111012  
> 4. 513  
> 5. 1A716

The [Decimal Number System](https://www.geeksforgeeks.org/digital-logic/decimal-number-system/) is the standard system for denoting numbers.

- It is also called the ****base-10**** system.
- Digits used in it are ****0, 1, 2, 3, 4, 5, 6, 7, 8, 9.****
- Each digit in the number is at a specific place value that is powers of 10.
- From right to left - units has the place value as 100, Tens has the place value as 101, Hundreds as 102, Thousands as 103, and so on.

****Example:****

> 10285 can be written as  
> 10285 = (1 × 104) + (0 × 103) + (2 × 102) + (8 × 101) + (5 × 100)  
> 10285 = 1 × 10000 + 0 × 1000 + 2 × 100 + 8 × 10+ 5 × 1  
> 10285 = 10000 + 0 + 200 + 80 + 5  
> 10285 = 10285

### Binary Number System

[Binary Number System](https://www.geeksforgeeks.org/maths/binary-number-system/) is the number system with ****base 2****.

- The numbers are formed using two digits - ****0 and 1.****
- Binary number system is very useful in electronic devices and computer systems because it can be easily performed using just two state i.e. 0 and 1. 
- Each digit in the number is at a specific place value that is powers of 2.
- From right to left - as powers of 2 i.e. 20, 21, 22, etc).

Binary Numbers can be converted to Decimal value by multiplying each digit with the place value and then adding the result.

****Example:****

> (1011)₂ can be written as  
> (1011)₂ = 1 × 2³ + 0 × 2² + 1 × 2¹ + 1 × 2⁰  
> (1011)₂ = 1 × 8 + 0 × 4 + 1 × 2 + 1 × 1  
> (1011)₂ = 11 (in decimal)

### Octal Number System

[Octal Number System](https://www.geeksforgeeks.org/maths/octal-number-system/) is the number system with ****base 8****.

- The numbers are formed using 8 digits i.e. ****0, 1, 2, 3, 4, 5, 6, 7.****
- Octal number system is useful for representing file permissions in Unix/Linux operating systems.
- Each digit in the number is at a specific place value that is powers of 8.
- From right to left - as powers of 8 i.e. 80, 81, 82, etc.

Octal Numbers can be converted to Decimal value by multiplying each digit with the place value and then adding the result.

****Example:**** 

> (325)8 can be written as  
> (325)8 = 3 × 8² + 2 × 8¹ + 5 × 8⁰  
> (325)8 = 192 + 16 + 5  
> (325)8 = 213 (in decimal)

### ****Hexadecimal Number System****

[Hexadecimal Number System](https://www.geeksforgeeks.org/maths/hexadecimal-number-system/) is the number system with ****base 16****.

- The numbers are formed using 16 digits i.e. ****0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E and F.****
- Hexadecimal Numbers are useful for handling memory address locations. 
- The digits from ****0 to 9**** are used as in the decimal system, but the numbers ****10 to 15**** are represented using the letters ****A to F**** as follows: 10 is represented as ****A****, 11 as ****B****, 12 as ****C****, 13 as ****D****, 14 as ****E****, 15 as ****F.****  
    ii. Place Value: the position of the digit. Each digit in the number is at a specific place value that is powers of 16. (from right to left - as powers of 16 i.e. 160, 161, 162, etc)

Hexadecimal Number System an be converted to Decimal value by multiplying each digit with the place value and then adding the result.

****Example:**** 

> (2F)16 can be written as  
> (2F)16 = 2 × 16¹ + F × 16⁰  
> (2F)16 = 2 × 16 + 15 × 1  
> (2F)16 = 32 + 15  
> (2F)16= 47 (in decimal)

### ****Also Check:****

> ****Conversion of Number Systems****
> 
> 1. [Decimal to Other Number Systems](https://www.geeksforgeeks.org/digital-logic/decimal-number-system/#:~:text=Conversion%20from%20Decimal%20to%20Other%20Number%20Systems)
> 2. [Binary to Other Number Systems](https://www.geeksforgeeks.org/maths/binary-number-system/#:~:text=Conversion%20from%20Binary%20to%20Other%20Number%20Systems)
> 3. [Octal to Other Number Systems](https://www.geeksforgeeks.org/maths/octal-number-system/#:~:text=Conversion%20from%20Octal%20to%20Other%20Number%20Systems)
> 4. [Hexadecimal to Other Number Systems](https://www.geeksforgeeks.org/maths/hexadecimal-number-system/#:~:text=Conversion%20from%20Hexadecimal%20to%20Other%20Number%20Systems)


The [number system](https://www.geeksforgeeks.org/maths/number-system-in-maths/) includes different types of numbers for example [prime numbers](https://www.geeksforgeeks.org/maths/prime-number/), [odd numbers](https://www.geeksforgeeks.org/maths/odd-numbers/), [even numbers](https://www.geeksforgeeks.org/maths/even-numbers/), [rational numbers](https://www.geeksforgeeks.org/maths/rational-numbers/), [whole numbers](https://www.geeksforgeeks.org/maths/whole-numbers/), etc. These numbers can be expressed in the form of figures as well as words accordingly. For example, numbers like 40 and 65 expressed in the form of figures can also be written as forty and sixty-five.

![number_system](https://media.geeksforgeeks.org/wp-content/uploads/20250404181615401782/number_system.webp)

Types of Number System

****Numbers Definition****

[Numbers](https://www.geeksforgeeks.org/maths/numbers/) are used in various arithmetic values applicable to carry out various arithmetic operations like addition, subtraction, multiplication, etc which are applicable in daily lives for the purpose of calculation. 

> ****Numbers**** are the mathematical values or figures used for the purpose of measuring or calculating quantities. It is represented by numerals as 2, 4, 7, etc. Some examples of numbers are integers, whole numbers, natural numbers, rational and irrational numbers, etc.

The value of a number is determined by the digit, its place value in the number, and the base of the number system. Numbers generally also known as numerals are the mathematical values used for counting, measurements, labeling, and measuring fundamental quantities.

## ****What is a Number System?****

> A [****Number system****](https://www.geeksforgeeks.org/maths/number-system-in-maths/) or ****numeral system**** is defined as an elementary system to express numbers and figures. It is the unique way of representing of numbers in arithmetic and algebraic structure.

Thus, in simple words, the writing system for denoting numbers using digits or symbols in a logical manner is defined as a ****Number system****. The numeral system Represents a useful set of numbers, reflects the arithmetic and [algebraic structure](https://www.geeksforgeeks.org/engineering-mathematics/groups-discrete-mathematics/) of a number, and provides standard representation. 

In the decimal number system, digits from 0 to 9 can be used to form all the numbers. With these digits, anyone can create infinite numbers. For example, 156,3907, 3456, 1298, 784859, etc. Other than digits, we can use alphabets such as A, B, C, D, E, and F (in Hexadecimal Number System) to represent different numbers.

## ****Types of Number Systems****

Based on the base value and the number of allowed digits, number systems are of many types. The four common types of Number systems are:

- ****Decimal Number System****
- ****Binary Number System****
- ****Octal Number System****
- ****Hexadecimal Number System****

## ****Decimal Number System****

A number system with a base value of 10 is termed a Decimal [number system](https://www.geeksforgeeks.org/maths/number-system-in-maths/). It uses 10 digits i.e. 0-9 for the creation of numbers. Here, each digit in the number is at a specific place with a place value of a product of different powers of 10. Here, the place value is termed from right to left as the first place value called units, second to the left as Tens, so on Hundreds, Thousands, etc. Here, units have a place value of 100, tens have a place value of 101, hundreds as 102, thousands as 103, and so on.

> ****For example, 12265 has place values as,****
> 
> (1 × 104) + (2 × 103) + (2 × 102) + (6 × 101) + (5 × 100)  
> = (1 × 10000) + (2 × 1000) + (2 × 100) + (6 × 10) + (5 × 1)  
> = 10000 + 2000 + 200 + 60 + 5  
> = 12265

## ****Binary Number System****

A number System with a base value of 2 is termed a Binary [number system](https://www.geeksforgeeks.org/maths/number-system-in-maths/). It uses 2 digits i.e. 0 and 1 for the creation of numbers. The numbers formed using these two digits are termed Binary Numbers. The binary number system is very useful in electronic devices and computer systems because it can be easily performed using just two states ON and OFF i.e. 0 and 1.

Decimal Numbers 0-9 are represented in binary as 0, 1, 10, 11, 100, 101, 110, 111, 1000, and 1001

****For example****, 14 can be written as 1110, 19 can be written as 10011, and 50 can be written as 110010.

> ****Example of 14 in the binary system****
> 
> ![frame_266](https://media.geeksforgeeks.org/wp-content/uploads/20250404181615533987/frame_266.webp)
> 
> 14 as Binary
> 
> ****Here 14 can be written as 1110****

## ****Octal Number System****

Octal Number System is one in which the base value is 8. It uses 8 digits i.e. 0-7 for the creation of Octal Numbers. Octal Numbers can be converted to Decimal values by multiplying each digit with the place value and then adding the result. Here the place values are 80, 81, and 82. Octal Numbers are useful for the representation of UTF8 Numbers. Example,

> (81)10 can be written as (121)8
> 
> (125)10 can be written as (175)8

## ****Hexadecimal Number System****

A number System with a base value of 16 is known as Hexadecimal Number System. It uses 16 digits for the creation of its numbers. Digits from 0-9 are taken like the digits in the decimal number system but the digits from 10-15 are represented as A-F i.e. 10 is represented as A, 11 as B, 12 as C, 13 as D, 14 as E, and 15 as F. Hexadecimal Numbers are useful for handling memory address locations. Examples,

> (185)10  can be written as (B9)16
> 
> (5440)10  can be written as (1540)16
> 
> (4265)10  can be written as (10A9)16
> 
> |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
> |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
> |****Hexadecimal****|****0****|****1****|****2****|****3****|****4****|****5****|****6****|****7****|****8****|****9****|****A****|****B****|****C****|****D****|****E****|****F****|
> |****Decimal****|****0****|****1****|****2****|****3****|****4****|****5****|****6****|****7****|****8****|****9****|****10****|****11****|****12****|****13****|****14****|****15****|

Other than these, one ancient number system which precedes the decimal [number system](https://www.geeksforgeeks.org/maths/number-system-in-maths/) is the Roman number system. Let's learn about it in detail as follows:

## ****Roman Number System****

The Roman numeral system is an ancient numerical system that was used in ancient Rome and throughout the Roman Empire. It is based on a combination of letters from the Latin alphabet to represent numbers. Here are the basic symbols used in the Roman numeral system along with their corresponding values:

|Symbol|Value|Symbol|Value|Symbol|Value|
|---|---|---|---|---|---|
|I|1|X|10|C|100|
|II|2|XX|20|CC|200|
|III|3|XXX|30|CCC|300|
|IV|4|XL|40|CD|400|
|V|5|L|50|D|500|
|VI|6|LX|60|DC|600|
|VII|7|LXX|70|DCC|700|
|VIII|8|LXXX|80|DCCC|800|
|IX|9|XC|90|CM|900|
|X|10|C|100|M|1000|

### Rule of Roman Numeral

To write numbers in Roman numerals, we can use the following rules:

- The symbols I, X, C, and M can be repeated up to three times in a row.
- When a smaller value symbol appears before a larger value symbol, the smaller value is subtracted. For example, IV represents 4 (5 - 1) and IX represents 9 (10 - 1).
- When a smaller value symbol appears after a larger value symbol, the smaller value is added. For example, VI represents 6 (5 + 1) and XI represents 11 (10 + 1).

## Sample Problems on Types of Number System

****Problem 1: Convert (4525)********8**** ****into a decimal.****

****Solution:****

> (4525)8 = 4 × 83 + 5 × 82 + 2 × 81 + 5 × 80
> 
> ⇒ 45258 = 4 × 512 + 5 × 64 + 2 × 8 + 5 × 1
> 
> ⇒ 45258 = 2048 + 320 + 16 + 5
> 
> ⇒ 45258 = 238910

****Problem 2: Convert (17)********10**** ****as a binary number.****

****Solution:****               

> ![frame_267](https://media.geeksforgeeks.org/wp-content/uploads/20250404181615580313/frame_267.webp)
> 
> 17 as Binary
> 
> Therefore (17)10 = (10001)2

****Problem 3: Convert (1011110)********2**** ****into an**** ****octal number.****

****Solution:****

> Given (1011110)2 a binary number, to convert it into octal number         
> 
> |Octal Number|Binary Number|
> |---|---|
> |0|000|
> |1|001|
> |2|010|
> |3|011|
> |4|100|
> |5|101|
> |6|110|
> |7|111|
> 
> Using this table we can write give number as 
> 
> 001 011 110 i .e  
> 001 = 1  
> 011 = 3  
> 110 = 6
> 
> So (1011110)2 in octal number is (136)8

****Problem 4: Convert the Roman numeral XLVIII into its decimal equivalent.****

****Solution:****

> As we know, all numbers before the greatest symbol are subtracted from it and all the numbers after the greatest symbol are added,
> 
> XLVIII = 50 - 10 + 5 + 1 + 1 + 1 = 48
> 
> ****Thus, XLVIII is 48 in decimal representation.****

****Problem 5: Convert the Roman numeral MCCXLV into its decimal equivalent.****

****Solution:****

> As we know, all numbers before the greatest symbol are subtracted from it and all the numbers after the greatest symbol are added,
> 
> MCCXLV = 1000 + 100 + 100 - 10 + 5  
> = 1000 + 200 + 5 - 10  
> = 1245
> 
> ****So, the Roman numeral MCCXLV is equal to 1245 in the decimal number system.****


- [Conversion between number systems](https://www.geeksforgeeks.org/digital-logic/number-system-and-base-conversions/)
- [Arithmetic Operations](https://www.geeksforgeeks.org/maths/arithmetic-operations/)
- [Modular Arithmetic](https://www.geeksforgeeks.org/engineering-mathematics/modular-arithmetic/)
- [Greatest Common Divisor](https://www.geeksforgeeks.org/maths/greatest-common-divisor-gcd/)
- [Congruency](https://www.geeksforgeeks.org/maths/congruency/)
- [Fermats Little Theorem](https://www.geeksforgeeks.org/dsa/fermats-little-theorem/)
- [Euclid's Division Algorithm](https://www.geeksforgeeks.org/maths/euclid-s-division-algorithm/)[](https://www.geeksforgeeks.org/maths/euclid-s-division-algorithm/)[](https://www.geeksforgeeks.org/maths/euclid-s-division-algorithm/)[](https://www.geeksforgeeks.org/maths/euclid-s-division-algorithm/)[](https://www.geeksforgeeks.org/maths/euclid-s-division-algorithm/)

## Combinatorics

Deals with counting, arrangement and discrete structures. Covers permutations, combinations, pigeonhole principle, inclusion-exclusion and recurrence relations used in algorithm design.

- [Basic Counting Rules](https://www.geeksforgeeks.org/maths/fundamental-principle-of-counting/)
- [Tree Diagram](https://www.geeksforgeeks.org/maths/tree-diagram-meaning-features-conditional-probability-and-examples/)
- [Permutation and Combination](https://www.geeksforgeeks.org/maths/permutations-and-combinations/)
- [Pigeonhole principle](https://www.geeksforgeeks.org/engineering-mathematics/discrete-mathematics-the-pigeonhole-principle/)
- [Inclusion-Exclusion principle](https://www.geeksforgeeks.org/maths/principle-of-inclusion-and-exclusion/)
- [Recurrence Relations](https://www.geeksforgeeks.org/dsa/recurrence-relations-a-complete-guide/)
- [Algorithms and Complexity](https://www.geeksforgeeks.org/computer-science-fundamentals/what-is-an-algorithm-definition-types-complexity-examples/)

## Discrete Mathematics

Studies logic, sets, functions and relations fundamental for data structures, algorithms and digital circuits. Induction is key for proving algorithm correctness.

- [Set Theory](https://www.geeksforgeeks.org/maths/set-theory/)
- [Propositional Logic](https://www.geeksforgeeks.org/engineering-mathematics/proposition-logic/)
- [Function in Mathematics](https://www.geeksforgeeks.org/maths/function-in-maths/)
- [Relations and Their Properties](https://www.geeksforgeeks.org/maths/relation-in-maths/)
- [Principle of Mathematical Induction](https://www.geeksforgeeks.org/maths/principle-of-mathematical-induction/)
- [Boolean Algebra](https://www.geeksforgeeks.org/digital-logic/boolean-algebra/)

## Linear Algebra

Explores vectors, matrices and transformations used in graphics, machine learning and data science. Includes eigenvalues, systems of equations and PCA.

- [Vector and Vector Spaces](https://www.geeksforgeeks.org/maths/vector-space/)
- [Matrices](https://www.geeksforgeeks.org/maths/introduction-to-matrices/)
- [Matrix Diagonalization](https://www.geeksforgeeks.org/dsa/matrix-diagonalization/)
- [Eigenvalues and Eigenvectors](https://www.geeksforgeeks.org/engineering-mathematics/eigen-values/)
- [System of linear equations](https://www.geeksforgeeks.org/engineering-mathematics/system-linear-equations/)
- [Gaussian Elimination to Solve Linear Equations](https://www.geeksforgeeks.org/dsa/gaussian-elimination/)
- [Principal Component Analysis](https://www.geeksforgeeks.org/machine-learning/mathematical-approach-to-pca/)

## Calculus

Used in optimization and modeling continuous systems. Covers limits, derivatives, integrals and differential equations relevant in algorithm analysis and simulations.

- [Limits, Continuity & Differentiation](https://www.geeksforgeeks.org/engineering-mathematics/limits-continuity-differentiability/)
- [Integration](https://www.geeksforgeeks.org/maths/integration/)
- [Partial Derivative](https://www.geeksforgeeks.org/engineering-mathematics/engineering-mathematics-partial-derivatives/)
- [Differential Equation](https://www.geeksforgeeks.org/maths/differential-equations/)

## Graph Theory

Mathematical study of graphs, their types and properties. Topics include paths, circuits, planarity and coloring key for modeling networks and relationships.

- [Graph Theory](https://www.geeksforgeeks.org/maths/mathematics-graph-theory-basics-set-1/)
- [Types of Graphs with Examples](https://www.geeksforgeeks.org/dsa/graph-types-and-applications/)
- [Graph Representations](https://www.geeksforgeeks.org/dsa/graph-and-its-representations/)
- [Walks, Trails, Paths, Cycles and Circuits](https://www.geeksforgeeks.org/engineering-mathematics/walks-trails-paths-cycles-and-circuits-in-graph/)
- [Planar Graphs and Graph Coloring](https://www.geeksforgeeks.org/engineering-mathematics/mathematics-planar-graphs-graph-coloring/)
- [Handshaking Lemma](https://www.geeksforgeeks.org/dsa/handshaking-lemma-and-interesting-tree-properties/)

## Probability and Statistics

Provides tools for analyzing uncertainty and data. Covers probability theory, distributions, Bayes’ theorem and statistical inference for machine learning and data modeling.

- [Probability Theory](https://www.geeksforgeeks.org/maths/probability-theory/)
- [Bayes' Theorem](https://www.geeksforgeeks.org/maths/bayes-theorem/)
- [Probability Distribution](https://www.geeksforgeeks.org/maths/probability-distribution/)
- [Descriptive Statistics](https://www.geeksforgeeks.org/data-science/descriptive-statistic/)
- [Sampling](https://www.geeksforgeeks.org/data-science/probability-sampling/)
- [Hypothesis Testing](https://www.geeksforgeeks.org/software-testing/understanding-hypothesis-testing/)
- [Regression Analysis](https://www.geeksforgeeks.org/machine-learning/what-is-regression-analysis/)

## Applications of Mathematics in Computer Science

1. [Algorithms and Data Structures](https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/): Used to design and analyze efficient algorithms using logic, combinatorics and complexity theory.
2. [Computer Graphics](https://www.geeksforgeeks.org/computer-graphics/introduction-to-computer-graphics/): Linear algebra and geometry help with 2D/3D rendering, animations and transformations.
3. [Machine Learning and AI](https://www.geeksforgeeks.org/machine-learning/ml-machine-learning/): Statistics, probability and calculus power model training, optimization and predictions.
4. [Cryptography](https://www.geeksforgeeks.org/computer-networks/cryptography-and-its-types/): Number theory and modular arithmetic secure data through encryption and hashing.
5. [Databases and Information Retrieval](https://www.geeksforgeeks.org/dbms/dbms/): Set theory and logic manage queries, relationships and indexing efficiently.
6. [Networks and Communication](https://www.geeksforgeeks.org/computer-networks/network-and-communication/): Graph theory models connections, routing and data flow in computer networks.
7. [Compiler Design and Automata](https://www.geeksforgeeks.org/compiler-design/introduction-of-compiler-design/): Formal languages and automata theory enable syntax analysis and language processing.

## ****Basic Problems****

- [Check Even or Odd](https://www.geeksforgeeks.org/dsa/check-whether-given-number-even-odd/)
- [Multiplication Table](https://www.geeksforgeeks.org/dsa/program-to-print-multiplication-table-of-a-number/)
- [Sum of Naturals](https://www.geeksforgeeks.org/dsa/program-find-sum-first-n-natural-numbers/)
- [Sum of Squares of Naturals](https://www.geeksforgeeks.org/dsa/sum-of-squares-of-first-n-natural-numbers/)
- [Swap Two Numbers](https://www.geeksforgeeks.org/dsa/swap-two-numbers/)
- [Closest Number](https://www.geeksforgeeks.org/dsa/find-number-closest-n-divisible-m/)
- [Dice Problem](https://www.geeksforgeeks.org/dsa/the-dice-problem/)
- [Nth Term of AP](https://www.geeksforgeeks.org/dsa/nth-term-of-ap-from-first-two-terms/)

## ****Easy Problems****

- [Sum of Digits](https://www.geeksforgeeks.org/dsa/program-for-sum-of-the-digits-of-a-given-number/)
- [Reverse Digits](https://www.geeksforgeeks.org/dsa/write-a-program-to-reverse-digits-of-a-number/)
- [Prime Testing](https://www.geeksforgeeks.org/dsa/introduction-to-primality-test-and-school-method/)
- [Check Power](https://www.geeksforgeeks.org/dsa/check-if-a-number-is-power-of-another-number/)
- [Distance between Two Points](https://www.geeksforgeeks.org/dsa/program-calculate-distance-two-points/)
- [Valid Triangle](https://www.geeksforgeeks.org/dsa/check-whether-triangle-valid-not-sides-given/)
- [Overlapping Rectangles](https://www.geeksforgeeks.org/dsa/find-two-rectangles-overlap/)
- [Factorial of a Number](https://www.geeksforgeeks.org/dsa/program-for-factorial-of-a-number/)
- [Pair Cube Count](https://www.geeksforgeeks.org/dsa/count-pairs-a-b-whose-sum-of-cubes-is-n-a3-b3-n/)
- [GCD or HCF](https://www.geeksforgeeks.org/dsa/program-to-find-gcd-or-hcf-of-two-numbers/)
- [LCM of Two Numbers](https://www.geeksforgeeks.org/dsa/program-to-find-lcm-of-two-numbers/)
- [Perfect Number](https://www.geeksforgeeks.org/dsa/perfect-number/)
- [Add Two Fraction](https://www.geeksforgeeks.org/dsa/program-to-add-two-fractions/)
- [Day of the Week](https://www.geeksforgeeks.org/dsa/find-day-of-the-week-for-a-given-date/)
- [Nth Fibonacci Number](https://www.geeksforgeeks.org/dsa/program-for-nth-fibonacci-number/)
- [Decimal to Binary](https://www.geeksforgeeks.org/dsa/program-decimal-binary-conversion/)
- [N-th term of 1, 3, 6, 10, 15, 21…](https://www.geeksforgeeks.org/dsa/find-nth-term-series-136101521/)
- [Armstrong Number](https://www.geeksforgeeks.org/dsa/program-for-armstrong-numbers/)
- [Palindrome Number](https://www.geeksforgeeks.org/dsa/check-if-a-number-is-palindrome/)
- [Digit Root](https://www.geeksforgeeks.org/dsa/digital-rootrepeated-digital-sum-given-integer/)

## ****Medium Problems****

- [Square Root](https://www.geeksforgeeks.org/dsa/square-root-of-an-integer/)
- [3 Divisors](https://www.geeksforgeeks.org/dsa/numbers-exactly-3-divisors/)
- [Divisible by 4](https://www.geeksforgeeks.org/dsa/check-large-number-divisible-4-not/)
- [Divisibility by 11](https://www.geeksforgeeks.org/dsa/check-large-number-divisible-11-not/)
- [Divisibility by 13](https://www.geeksforgeeks.org/dsa/check-large-number-divisible-13-not/)
- [K-th Digit in a^b](https://www.geeksforgeeks.org/dsa/k-th-digit-raised-power-b/)
- [Fraction to Recurring Decimal](https://www.geeksforgeeks.org/dsa/represent-the-fraction-of-two-numbers-in-the-string-format/)
- [Recurring Sequence in a Fraction](https://www.geeksforgeeks.org/dsa/find-recurring-sequence-fraction/)
- [Compute nPr](https://www.geeksforgeeks.org/dsa/program-to-calculate-the-value-of-npr/)
- [Compute nCr](https://www.geeksforgeeks.org/dsa/program-calculate-value-ncr/)
- [Pascal’s Triangle](https://www.geeksforgeeks.org/dsa/pascal-triangle/)
- [All Factor (Or Divisors)](https://www.geeksforgeeks.org/dsa/find-all-factors-of-a-natural-number/)
- [Prime Factorization](https://www.geeksforgeeks.org/dsa/prime-factor/)
- [Largest Prime factor](https://www.geeksforgeeks.org/dsa/find-largest-prime-factor-number/)
- [Modular Exponentiation](https://www.geeksforgeeks.org/dsa/modular-exponentiation-power-in-modular-arithmetic/)
- [nth Catalan Number](https://www.geeksforgeeks.org/dsa/program-nth-catalan-number/)
- [Binomial Coefficient](https://www.geeksforgeeks.org/dsa/binomial-coefficient-dp-9/)
- [Power Set](https://www.geeksforgeeks.org/dsa/power-set/)
- [Next Permutation](https://www.geeksforgeeks.org/dsa/next-permutation/)

## ****Hard Problems****

- [Sieve of Eratosthenes](https://www.geeksforgeeks.org/dsa/sieve-of-eratosthenes/)
- [Clock Angle Problem](https://www.geeksforgeeks.org/dsa/calculate-angle-hour-hand-minute-hand/)
- [Tower of Hanoi](https://www.geeksforgeeks.org/dsa/c-program-for-tower-of-hanoi/)
- [Rat and Poisoned](https://www.geeksforgeeks.org/dsa/rat-and-poisoned-bottle-problem/)
- [8 puzzle Problem](https://www.geeksforgeeks.org/dsa/8-puzzle-problem-using-branch-and-bound/)
- [Determinant of a Matrix](https://www.geeksforgeeks.org/dsa/determinant-of-a-matrix/)
- [Euler's Totient Function](https://www.geeksforgeeks.org/dsa/eulers-totient-function/)
- [Josephus Problem](https://www.geeksforgeeks.org/dsa/josephus-problem/)

****Recommended Links****

- [Top Logic Building Interview Problems](https://www.geeksforgeeks.org/dsa/top-problems-on-logic-building-problems-for-interviews/)
- [Practice Logic Building Problems](https://www.geeksforgeeks.org/explore)
- [Pattern Printing Problems](https://www.geeksforgeeks.org/dsa/pattern-printing-problems/)
- [Mathematical Algorithms](https://www.geeksforgeeks.org/dsa/mathematical-algorithms-difficulty-wise/)


- [Quiz on Logic Building](https://www.geeksforgeeks.org/quizzes/dsa-tutorial-logic-building/)

### 2. Learn about Complexities

To analyze algorithms, we mainly measure order of growth of time or space taken in terms of input size. We do this in the worst case scenario in most of the cases. Please refer the below links for a clear understanding of these concepts.

- [Complexity Analysis Guide](https://www.geeksforgeeks.org/dsa/analysis-of-algorithms/)
- [Quiz on Complexity Analysis](https://www.geeksforgeeks.org/quizzes/quiz-on-complexity-analysis-for-dsa/)

### 3. Array

****Array**** is a linear data structure where elements are allocated ****contiguous memory****, allowing for ****constant-time access****.[](https://www.geeksforgeeks.org/dsa/introduction-to-arrays-data-structure-and-algorithm-tutorials/)

- [Array Guide](https://www.geeksforgeeks.org/dsa/array-data-structure-guide/)
- [Quiz on Arrays](https://www.geeksforgeeks.org/quizzes/dsa-tutorial-array/)

### 4. Searching Algorithms

****Searching algorithms**** are used to locate specific data within a large set of data. It helps ****find a target value**** within the data. There are various types of searching algorithms, each with its own approach and efficiency.

- [Searching Guide](https://www.geeksforgeeks.org/dsa/searching-algorithms/)
- [Quiz on Searching](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-searching-algorithm-with-answers/)

### 5. Sorting Algorithm

****Sorting algorithms**** are used to ****arrange**** the elements of a list in a ****specific order****, such as numerical or alphabetical. It organizes the items in a systematic way, making it easier to search for and access specific elements.

- [Sorting Guide](https://www.geeksforgeeks.org/dsa/sorting-algorithms/)
- [Quiz on Sorting](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-sorting-algorithms-with-answers/)

### 6. Hashing

Hashing is a technique that generates a fixed-size output (hash value) from an input of variable size using mathematical formulas called hash functions. Hashing is commonly used in data structures for efficient searching, insertion and deletion.

- [Hashing Guide](https://www.geeksforgeeks.org/dsa/hashing-data-structure/)
- [Quiz on Hashing](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-hash-data-strcuture-with-answers/)

### 7. Two Pointer Technique

****I****n Two Pointer Technique, we typically use two index variables from two corners of an array. We use the two pointer technique for searching a required point or value in an array.

- [Two Pointer Technique](https://www.geeksforgeeks.org/dsa/two-pointers-technique/)
- [Quiz on Two Pointer Technique](https://www.geeksforgeeks.org/quizzes/quiz-on-two-pointer-technique-for-dsa/)

### 8. Window Sliding Technique

****I****n Window Sliding Technique, we use the result of previous subarray to quickly compute the result of current.

- [Window Sliding Technique](https://www.geeksforgeeks.org/dsa/window-sliding-technique/)
- [Quiz on Sliding Window](https://www.geeksforgeeks.org/quizzes/quiz-on-sliding-window-technique-for-dsa/)

### 9. Prefix Sum Technique

****I****n Prefix Sum Technique, we compute prefix sums of an array to quickly find results for a subarray.

- [Prefix Sum Technique](https://www.geeksforgeeks.org/dsa/prefix-sum-array-implementation-applications-competitive-programming/)
- [Quiz on Prefix Sum](https://www.geeksforgeeks.org/quizzes/quiz-on-prefix-sum-for-dsa/)

### 10. String

****String**** is a sequence of characters, typically immutable and have limited set of elements (lower case or all English alphabets).

- [Strings Guide](https://www.geeksforgeeks.org/dsa/string-data-structure/)
- [Quiz on Strings](https://www.geeksforgeeks.org/quizzes/quiz-on-string-for-dsa/)

### 11. Recursion

****Recursion**** is a programming technique where a function ****calls itself**** within its own definition. It is usually used to solve problems that can be broken down into smaller instances of the same problem.

- [Recursion Guide](https://www.geeksforgeeks.org/dsa/recursion-algorithms/)
- [Quiz on Recursion](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-recursion-algorithm-with-answers/)

### 12. Matrix/Grid

****Matrix**** is a two-dimensional array of elements, arranged in ****rows**** and ****columns****. It is represented as a rectangular grid, with each element at the intersection of a row and column.

- [Matrix Guide](https://www.geeksforgeeks.org/dsa/matrix/)
- [Quiz on Matrix/Grid.](https://www.geeksforgeeks.org/quizzes/quiz-on-matrixgrid-for-dsa/)

### 13. Linked List

****Linked list**** is a linear data structure that stores data in nodes, which are connected by pointers. Unlike arrays, nodes of linked lists are not stored in contiguous memory locations and can only be ****accessed sequentially****, starting from the head of list.

- [Linked List Guide](https://www.geeksforgeeks.org/dsa/linked-list-data-structure/)
- [Quiz on Linked List](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-linked-list-data-structure-with-answers/)

### 14. Stack

****Stack**** is a linear data structure that follows the ****Last In, First Out (LIFO)**** principle. Stacks play an important role in managing function calls, memory, and are widely used in algorithms like stock span problem, next greater element and largest area in a histogram.

- [Stack Guide](https://www.geeksforgeeks.org/dsa/stack-data-structure/)
- [Quiz on Stack](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-stack-data-strcuture-with-answers/)

### 15. Queue

****Queue**** is a linear data structure that follows the ****First In, First Out (FIFO)**** principle. Queues play an important role in managing tasks or data in order, scheduling and message handling systems.

- [Queue Guide](https://www.geeksforgeeks.org/dsa/queue-data-structure/)
- [Quiz on Queue](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-queue-data-structure-with-answers/)

### 16. Deque

A ****deque**** (double-ended queue) is a data structure that allows elements to be added or removed from both ends efficiently.

- [Deque Guide](https://www.geeksforgeeks.org/dsa/deque-set-1-introduction-applications/)
- [Quiz on Deque](https://www.geeksforgeeks.org/quizzes/deque-960/)

### 17. Tree

****Tree**** is a ****non-linear, hierarchical**** data structure consisting of nodes connected by edges, with a top node called the ****root**** and nodes having child nodes. It is widely used in ****file systems****, ****databases****, ****decision-making algorithms****, etc.

- [Tree Guide](https://www.geeksforgeeks.org/dsa/tree-data-structure/)
- [Quiz on Tree](https://www.geeksforgeeks.org/quizzes/tree-22648/)

### 18. Heap

****Heap**** is a ****complete binary tree**** data structure that satisfies the ****heap property****. Heaps are usually used to implement [priority queues](https://www.geeksforgeeks.org/dsa/priority-queue-set-1-introduction/), where the ****smallest**** or ****largest**** element is always at the root of the tree.

- [Heap Guide](https://www.geeksforgeeks.org/dsa/heap-data-structure/)
- [Quiz on Heap](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-heap-data-strcuture-with-answers/)

### 19. Graph

****Graph**** is a ****non-linear**** data structure consisting of a finite set of ****vertices****(or nodes) and a set of ****edges****(or links)that connect a pair of nodes. Graphs are widely used to represent relationships between entities.

- [Graph Guide](https://www.geeksforgeeks.org/dsa/graph-data-structure-and-algorithms/)
- [Quiz on Graph](https://www.geeksforgeeks.org/quizzes/graph-12715/)

### 20. Greedy Algorithm

****Greedy Algorithm**** builds up the solution one piece at a time and chooses the next piece which gives the most obvious and immediate benefit i.e., which is the most ****optimal choice at that moment****. So the problems where choosing ****locally optimal**** also leads to the global solutions are best fit for Greedy.

- [Greedy Algorithms Guide](https://www.geeksforgeeks.org/dsa/greedy-algorithms/)
- [Quiz on Greedy](https://www.geeksforgeeks.org/quizzes/)

### 21. Dynamic Programming

****Dynamic Programming**** is a method used to solve complex problems by breaking them down into simpler ****subproblems****. By solving each subproblem only ****once**** and ****storing the results****, it avoids redundant computations, leading to more ****efficient solutions**** for a wide range of problems.

- [Dynamic Programming Guide](https://www.geeksforgeeks.org/competitive-programming/dynamic-programming/)
- [Quiz on DP](https://www.geeksforgeeks.org/quizzes/)

### 22. Advanced Data Structure and Algorithms

Advanced Data Structures like ****Trie****, ****Segment Tree****, ****Red-Black Tree**** and ****Binary Indexed Tree**** offer significant performance improvements for specific problem domains. They provide efficient solutions for tasks like fast prefix searches, range queries, dynamic updates, and maintaining balanced data structures, which are crucial for handling large datasets and real-time processing.

- [Trie](https://www.geeksforgeeks.org/dsa/introduction-to-trie-data-structure-and-algorithm-tutorials/)
- [Segment Tree](https://www.geeksforgeeks.org/dsa/segment-tree-data-structure/)
- [Red-Black Tree](https://www.geeksforgeeks.org/dsa/introduction-to-red-black-tree/)
- [Binary Indexed Tree](https://www.geeksforgeeks.org/dsa/binary-indexed-tree-or-fenwick-tree-2/)
- [Practice Advanced Data Structures](https://www.geeksforgeeks.org/dsa/advance-data-structure/)

### 23. Other Algorithms

****Bitwise Algorithms:**** Operate on individual bits of numbers.

- [Bitwise Algorithms Guide](https://www.geeksforgeeks.org/dsa/bitwise-algorithms/)
- [Quiz on Bit Magic](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-bitwise-algorithms-and-bit-manipulations-with-answers/)

****Backtracking Algorithm :**** Follow Recursion with the option to ****revert and traces back**** if the solution from current point is not feasible.

- [Backtracking Guide](https://www.geeksforgeeks.org/dsa/backtracking-algorithms/)[](https://www.geeksforgeeks.org/dsa/backtracking-algorithms/)
- [Quiz on Backtracking](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-backtracking-algorithm-with-answers/)

****Divide and conquer:**** A strategy to solve problems by dividing them into ****smaller subproblems****, solving those subproblems, and combining the solutions to obtain the final solution.

- [Divide and Conquer Guide](https://www.geeksforgeeks.org/dsa/divide-and-conquer/)
- [Quiz on Divide and Conquer](https://www.geeksforgeeks.org/quizzes/top-mcqs-on-divide-and-conquer-algrithm-with-answers/)

****Branch and Bound :**** Used in combinatorial optimization problems to systematically search for the best solution. It works by dividing the problem into smaller subproblems, or branches, and then eliminating certain branches based on bounds on the optimal solution. This process continues until the best solution is found or all branches have been explored.

- [Branch and Bound Algorithm](https://www.geeksforgeeks.org/dsa/branch-and-bound-algorithm/)

****Geometric algorithms**** are a set of algorithms that solve problems related to ****shapes****, ****points****, ****lines**** and polygons.

- [Geometric Algorithms](https://www.geeksforgeeks.org/dsa/geometric-algorithms/)
- [Practice Geometric Algorithms](https://www.geeksforgeeks.org/explore?page=1&category=Geometric&sortBy=submissions)

****Randomized algorithms**** are algorithms that use ****randomness**** to solve problems. They make use of random input to achieve their goals, often leading to ****simpler**** and more ****efficient solutions****. These algorithms may ****not product same result**** but are particularly useful in situations when a ****probabilistic approach**** is acceptable.

- [Randomized Algorithms](https://www.geeksforgeeks.org/dsa/randomized-algorithms/)

  

Comment

More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/dsa/analysis-of-algorithms) 

[Analysis of Algorithms](https://www.geeksforgeeks.org/dsa/analysis-of-algorithms)

### Similar Reads

## Basics & Prerequisites

## Data Structures

## Algorithms

## Advanced

## Interview Preparation

## Practice Problem

[![geeksforgeeks-footer-logo](https://media.geeksforgeeks.org/auth-dashboard-uploads/gfgFooterLogo.png)](https://www.geeksforgeeks.org/)

Corporate & Communications Address:

A-143, 7th Floor, Sovereign Corporate Tower, Sector- 136, Noida, Uttar Pradesh (201305)

Registered Address:

K 061, Tower K, Gulshan Vivante Apartment, Sector 137, Noida, Gautam Buddh Nagar, Uttar Pradesh, 201305

[

](https://www.facebook.com/geeksforgeeks.org/)[

](https://www.instagram.com/geeks_for_geeks/)[

](https://in.linkedin.com/company/geeksforgeeks)[

](https://twitter.com/geeksforgeeks)[

](https://www.youtube.com/geeksforgeeksvideos)

[![GFG App on Play Store](https://media.geeksforgeeks.org/auth-dashboard-uploads/googleplay.png)](https://geeksforgeeksapp.page.link/gfg-app)[![GFG App on App Store](https://media.geeksforgeeks.org/auth-dashboard-uploads/appstore.png)](https://geeksforgeeksapp.page.link/gfg-app)

[Advertise with us](https://www.geeksforgeeks.org/advertise-with-us/)

- Company
- [About Us](https://www.geeksforgeeks.org/about/)
- [Legal](https://www.geeksforgeeks.org/legal/)
- [Privacy Policy](https://www.geeksforgeeks.org/legal/privacy-policy/)
- [In Media](https://www.geeksforgeeks.org/press-release/)
- [Contact Us](https://www.geeksforgeeks.org/about/contact-us/)
- [Advertise with us](https://www.geeksforgeeks.org/advertise-with-us/)
- [GFG Corporate Solution](https://www.geeksforgeeks.org/gfg-corporate-solution/)
- [Placement Training Program](https://www.geeksforgeeks.org/campus-training-program/)

- [Languages](https://www.geeksforgeeks.org/introduction-to-programming-languages/)
- [Python](https://www.geeksforgeeks.org/python-programming-language/)
- [Java](https://www.geeksforgeeks.org/java/)
- [C++](https://www.geeksforgeeks.org/c-plus-plus/)
- [PHP](https://www.geeksforgeeks.org/php-tutorials/)
- [GoLang](https://www.geeksforgeeks.org/golang/)
- [SQL](https://www.geeksforgeeks.org/sql-tutorial/)
- [R Language](https://www.geeksforgeeks.org/r-tutorial/)
- [Android Tutorial](https://www.geeksforgeeks.org/android-tutorial/)
- [Tutorials Archive](https://www.geeksforgeeks.org/geeksforgeeks-online-tutorials-free/)

- [DSA](https://www.geeksforgeeks.org/learn-data-structures-and-algorithms-dsa-tutorial/)
- [DSA Tutorial](https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/)
- [Basic DSA Problems](https://www.geeksforgeeks.org/basic-coding-problems-in-dsa-for-beginners/)
- [DSA Roadmap](https://www.geeksforgeeks.org/complete-roadmap-to-learn-dsa-from-scratch/)
- [Top 100 DSA Interview Problems](https://www.geeksforgeeks.org/top-100-data-structure-and-algorithms-dsa-interview-questions-topic-wise/)
- [DSA Roadmap by Sandeep Jain](https://www.geeksforgeeks.org/dsa-roadmap-for-beginner-to-advanced-by-sandeep-jain/)
- [All Cheat Sheets](https://www.geeksforgeeks.org/geeksforgeeks-master-sheet-list-of-all-cheat-sheets/)

- [Data Science & ML](https://www.geeksforgeeks.org/ai-ml-ds/)
- [Data Science With Python](https://www.geeksforgeeks.org/data-science-tutorial/)
- [Data Science For Beginner](https://www.geeksforgeeks.org/data-science-for-beginners/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [ML Maths](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Data Visualisation](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [Pandas](https://www.geeksforgeeks.org/pandas-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)

- [Web Technologies](https://www.geeksforgeeks.org/web-technology/)
- [HTML](https://www.geeksforgeeks.org/html/)
- [CSS](https://www.geeksforgeeks.org/css/)
- [JavaScript](https://www.geeksforgeeks.org/javascript/)
- [TypeScript](https://www.geeksforgeeks.org/typescript/)
- [ReactJS](https://www.geeksforgeeks.org/learn-reactjs/)
- [NextJS](https://www.geeksforgeeks.org/nextjs/)
- [Bootstrap](https://www.geeksforgeeks.org/bootstrap/)
- [Web Design](https://www.geeksforgeeks.org/web-design/)

- [Python Tutorial](https://www.geeksforgeeks.org/python-programming-language/)
- [Python Programming Examples](https://www.geeksforgeeks.org/python-programming-examples/)
- [Python Projects](https://www.geeksforgeeks.org/python-projects-beginner-to-advanced/)
- [Python Tkinter](https://www.geeksforgeeks.org/python-tkinter-tutorial/)
- [Python Web Scraping](https://www.geeksforgeeks.org/python-web-scraping-tutorial/)
- [OpenCV Tutorial](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [Python Interview Question](https://www.geeksforgeeks.org/python-interview-questions/)
- [Django](https://www.geeksforgeeks.org/django-tutorial/)

- Computer Science
- [Operating Systems](https://www.geeksforgeeks.org/operating-systems/)
- [Computer Network](https://www.geeksforgeeks.org/computer-network-tutorials/)
- [Database Management System](https://www.geeksforgeeks.org/dbms/)
- [Software Engineering](https://www.geeksforgeeks.org/software-engineering/)
- [Digital Logic Design](https://www.geeksforgeeks.org/digital-electronics-logic-design-tutorials/)
- [Engineering Maths](https://www.geeksforgeeks.org/engineering-mathematics-tutorials/)
- [Software Development](https://www.geeksforgeeks.org/software-development/)
- [Software Testing](https://www.geeksforgeeks.org/software-testing-tutorial/)

- [DevOps](https://www.geeksforgeeks.org/devops-tutorial/)
- [Git](https://www.geeksforgeeks.org/git-tutorial/)
- [Linux](https://www.geeksforgeeks.org/linux-tutorial/)
- [AWS](https://www.geeksforgeeks.org/aws-tutorial/)
- [Docker](https://www.geeksforgeeks.org/docker-tutorial/)
- [Kubernetes](https://www.geeksforgeeks.org/kubernetes-tutorial/)
- [Azure](https://www.geeksforgeeks.org/microsoft-azure/)
- [GCP](https://www.geeksforgeeks.org/google-cloud-platform-tutorial/)
- [DevOps Roadmap](https://www.geeksforgeeks.org/devops-roadmap/)

- [System Design](https://www.geeksforgeeks.org/system-design-tutorial/)
- [High Level Design](https://www.geeksforgeeks.org/what-is-high-level-design-learn-system-design/)
- [Low Level Design](https://www.geeksforgeeks.org/what-is-low-level-design-or-lld-learn-system-design/)
- [UML Diagrams](https://www.geeksforgeeks.org/unified-modeling-language-uml-introduction/)
- [Interview Guide](https://www.geeksforgeeks.org/system-design-interview-guide/)
- [Design Patterns](https://www.geeksforgeeks.org/software-design-patterns/)
- [OOAD](https://www.geeksforgeeks.org/object-oriented-analysis-and-design/)
- [System Design Bootcamp](https://www.geeksforgeeks.org/system-design-interview-bootcamp-guide/)
- [Interview Questions](https://www.geeksforgeeks.org/most-commonly-asked-system-design-interview-problems-questions/)

- [Inteview Preparation](https://www.geeksforgeeks.org/technical-interview-preparation/)
- [Competitive Programming](https://www.geeksforgeeks.org/competitive-programming-a-complete-guide/)
- [Top DS or Algo for CP](https://www.geeksforgeeks.org/top-algorithms-and-data-structures-for-competitive-programming/)
- [Company-Wise Recruitment Process](https://www.geeksforgeeks.org/company-wise-recruitment-process/)
- [Company-Wise Preparation](https://www.geeksforgeeks.org/company-preparation/)
- [Aptitude Preparation](https://www.geeksforgeeks.org/aptitude-questions-and-answers/)
- [Puzzles](https://www.geeksforgeeks.org/puzzles/)

- School Subjects
- [Mathematics](https://www.geeksforgeeks.org/maths/)
- [Physics](https://www.geeksforgeeks.org/physics/)
- [Chemistry](https://www.geeksforgeeks.org/chemistry/)
- [Biology](https://www.geeksforgeeks.org/biology/)
- [Social Science](https://www.geeksforgeeks.org/social-science/)
- [English Grammar](https://www.geeksforgeeks.org/english-grammar/)
- [Commerce](https://www.geeksforgeeks.org/commerce/)

- [GeeksforGeeks Videos](https://www.geeksforgeeks.org/videos/)
- [DSA](https://www.geeksforgeeks.org/videos/category/sde-sheet/)
- [Python](https://www.geeksforgeeks.org/videos/category/python/)
- [Java](https://www.geeksforgeeks.org/videos/category/java-w6y5f4/)
- [C++](https://www.geeksforgeeks.org/videos/category/c/)
- [Web Development](https://www.geeksforgeeks.org/videos/category/web-development/)
- [Data Science](https://www.geeksforgeeks.org/videos/category/data-science/)
- [CS Subjects](https://www.geeksforgeeks.org/videos/category/cs-subjects/)

[@GeeksforGeeks, Sanchhaya Education Private Limited](https://www.geeksforgeeks.org/), [All rights reserved](https://www.geeksforgeeks.org/copyright-information/)

We use cookies to ensure you have the best browsing experience on our website. By using our site, you acknowledge that you have read and understood our [Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) & [Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)Got It !

![Lightbox](https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/)