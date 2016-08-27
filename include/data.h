#ifndef NN_DATA_DATA_H_
#define NN_DATA_DATA_H_

#include <memory>

#include "tensor/vector.h"
#include "tensor/matrix.h"

namespace nn {

class Data {
public:
  virtual ~Data() {}
};

using DataPtr = std::shared_ptr<Data>;


template <typename Tp>
class ScalarData : public Data {
public:
  ScalarData() : val_(Tp()) {}
  ScalarData(const ScalarData& sca_init) : val_(sca_init.val_) {}
  /*explicit */ScalarData(const Tp& val_init) : val_(val_init) {}
  ScalarData(ScalarData&& sca_init) : val_(std::move(sca_init.val_)) {}
  ScalarData& operator=(const ScalarData& sca_copy) {
    val_ = sca_copy.val_;
    return *this;
  }
  ScalarData& operator=(const Tp& val_copy) {
    val_ = val_copy;
    return *this;
  }
  ScalarData& operator=(ScalarData&& sca_move) {
    val_ = std::move(sca_move.val_);
    return *this;
  }
  ~ScalarData() {}
  
  Tp& operator()() { return val_; }
  const Tp& operator()() const { return val_; }

private:
  Tp val_;
};

template <typename Tp>
using ScalarDataPtr = std::shared_ptr<ScalarData<Tp>>;


template <typename Tp>
class VectorData : public Data {
public:
  VectorData() {}
  VectorData(const VectorData& vd_init) : vec_(vd_init.vec_) {}
  /*explicit */VectorData(const tensor::Vector<Tp>& vec_init) : vec_(vec_init) {}
  VectorData(VectorData&& vd_init) : vec_(std::move(vd_init.vec_)) {}
  VectorData& operator=(const VectorData& vd_copy) {
    vec_ = vd_copy.vec_;
    return *this;
  }
  VectorData& operator=(const tensor::Vector<Tp>& vec_copy) {
    vec_ = vec_copy;
    return *this;
  }
  VectorData& operator=(VectorData&& vd_move) {
    vec_ = std::move(vd_move.vec_);
    return *this;
  }
  ~VectorData() {}

  tensor::Vector<Tp>& operator()() { return vec_; }
  const tensor::Vector<Tp>& operator()() const { return vec_; }

private:
  tensor::Vector<Tp> vec_;
};


template <typename Tp>
using VectorDataPtr = std::shared_ptr<VectorData<Tp>>;


template <typename Tp>
class MatrixData : public Data {
public:
  MatrixData() {}
  MatrixData(const MatrixData& md_init) : mat_(md_init.mat_) {}
  explicit MatrixData(const tensor::Matrix<Tp>& mat_init) : mat_(mat_init) {}
  MatrixData(MatrixData&& md_init) : mat_(std::move(md_init.mat_)) {}
  MatrixData& operator=(const MatrixData& md_copy) {
    mat_ = md_copy.mat_;
    return *this;
  }
  MatrixData& operator=(const tensor::Matrix<Tp>& mat_copy) {
    mat_ = mat_copy;
    return *this;
  }
  MatrixData& operator=(MatrixData&& md_move) {
    mat_ = std::move(md_move.mat_);
    return *this;
  }
  ~MatrixData() {}

  tensor::Matrix<Tp>& operator()() { return mat_; }
  const tensor::Matrix<Tp>& operator()() const { return mat_; }

private:
  tensor::Matrix<Tp> mat_;
};

template <typename Tp>
using MatrixDataPtr = std::shared_ptr<MatrixData<Tp>>;


}  // namespace nn

#endif  // NN_DATA_DATA_H_