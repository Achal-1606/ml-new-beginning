## Eigen Values and Eigen Vectors

* Basis Vectors
    - Given a transformation T, such that it flipped the vector
    - There are some vectors which are not get change in direction but 
      will only get scaled up (the line they "span" won't change).
    - These type of vectors are called **Eigen Vectors**, having a transformation such as
    - __T(v) => A.v = k.v, so transformation is only in change of scales__
* Eigen Values
    - ```=> A.v = k.v```<br>```=> A.v - k.v = 0```<br>```=> v ( A.I - k ) = 0```
    - As v != 0, __NullSpace( A.I - k ) is non trivial__(Zero vector is not the only member)
    - Explanation of above point
        + For a matrix D, <br>
            D's cols are Linearly independent iff NullSpace(D) = {0} <br>
            But since the above equation NullSpace doesn't consist of 0, so... <br>
            __(A.I - k)  -> must have Linearly Dependent cols -> not invertible -> det. = 0__
        + So, for __A.v = k.v__ for non zero v's iff __det(A.I - k) = 0__
        + Above determinant is used to find the values of 'k' which will give us the Eigen values.
* Eigen Vectors
    - For any Eigen Value (k), the Eigen Vectors (Eigen Space, comprising of all the Eigen vectors)
    - E[k] = NullSpace(k.I - A)
    - if k = 5 <br>
      E[5] = NullSpace(5.I - A.I) = Eigen space corresponding to Eigen Value= 5 <br>=> __Reduced Row Echillon form__ of the matrix <br>=> Calculate the span for that will be the Eigen Vector and any transformation on this line will be transformed by Eigen Value.

