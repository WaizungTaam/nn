#include "../include/data.h"

#include "unittest.h"
#include "../include/tensor/vector.h"
#include "../include/tensor/matrix.h"

using namespace unittest;


class DataTest : public UnitTest {
protected:
  void setup() {}
  void teardown() {}
};

class Data_t : public DataTest {};
class ScalarData_t : public DataTest {};
class VectorData_t : public DataTest {};
class MatrixData_t : public DataTest {};


TEST_F(Data_t, All) {

}

TEST_F(ScalarData_t, constructors) {
  nn::ScalarData<int> s_i_1;
  nn::ScalarData<int> s_i_2 = 1;
  nn::ScalarData<int> s_i_3 = s_i_2;
  nn::ScalarData<int> s_i_4 = nn::ScalarData<int>(2);
  s_i_3 = s_i_2;
  s_i_4 = 3;
  s_i_1 = nn::ScalarData<int>(4);
}
TEST_F(ScalarData_t, op_parenthese) {
  nn::ScalarData<int> s_i = 1;
  EXPECT_EQ(s_i(), 1);
  nn::ScalarData<double> s_d = 1.2;
  EXPECT_EQ(s_d(), 1.2);
  const nn::ScalarData<int> s_i_1 = 2;
  EXPECT_EQ(s_i_1(), 2);
  s_i() = 3;
  EXPECT_EQ(s_i(), 3);
}

TEST_F(VectorData_t, constructors) {
  nn::VectorData<int> v_i;
  nn::VectorData<int> v_i_1 = tensor::Vector<int>({1, 2, 3, 4});
  nn::VectorData<int> v_i_2 = v_i_1;
  nn::VectorData<int> v_i_3 = nn::VectorData<int>();
  v_i_3 = v_i_2;
  v_i_2 = tensor::Vector<int>({1, 2});
  v_i_1 = nn::VectorData<int>();
}
TEST_F(VectorData_t, op_parenthese) {
  tensor::Vector<int> vec_i = {1, 2, 3, 4};
  tensor::Vector<int> vec_i_1 = {1, 2};
  tensor::Vector<double> vec_d = {1.2, 3.4};

  nn::VectorData<int> v_i = vec_i;
  EXPECT_TRUE(v_i().equal(vec_i));
  v_i() = vec_i_1;
  EXPECT_TRUE(v_i().equal(vec_i_1));

  const nn::VectorData<int> v_i_1 = vec_i;
  EXPECT_TRUE(v_i_1().equal(vec_i));

  nn::VectorData<double> v_d = vec_d;
  EXPECT_TRUE(v_d().equal(vec_d));
}

TEST_F(MatrixData_t, constructors) {

}
TEST_F(MatrixData_t, op_parenthese) {

}


int main() {
  RUN_ALL_TESTS();
}