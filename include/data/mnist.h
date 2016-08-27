#ifndef NN_DATA_MINST_H_
#define NN_DATA_MINST_H_

#include <fstream>
#include <string>
#include <vector>
#include "../tensor/vector.h"
#include "../tensor/matrix.h"

namespace nn {
namespace data {

class MNIST {
public:
  MNIST() : path_("") {}
  MNIST(const std::string& path_init) : path_(path_init) {}
  void load_train(tensor::Matrix<double>& img_data,
                  tensor::Matrix<double>& lbl_data) const {
    load_img_lbl(img_data, lbl_data, FILE_IMG_TRAIN, FILE_LBL_TRAIN);
  }
  void load_test(tensor::Matrix<double>& img_data,
                 tensor::Matrix<double>& lbl_data) const {
    load_img_lbl(img_data, lbl_data, FILE_IMG_TEST, FILE_LBL_TEST);
  }

private:
  struct ImgHeader {
    unsigned char magic_number[4];
    unsigned char num_imgs[4];
    unsigned char num_rows[4];
    unsigned char num_cols[4];
  };
  struct LblHeader {
    unsigned char magic_number[4];
    unsigned char num_lbls[4];
  };

  const int IMG_MAGIC_NUMBER = 2051;
  const int LBL_MAGIC_NUMBER = 2049;
  const std::string FILE_IMG_TRAIN = "train-images.idx3-ubyte";
  const std::string FILE_LBL_TRAIN = "train-labels.idx1-ubyte";
  const std::string FILE_IMG_TEST = "t10k-images.idx3-ubyte";
  const std::string FILE_LBL_TEST = "t10k-labels.idx1-ubyte";

  std::string path_;

  int uchar_to_int(unsigned char* uchar_arr, std::size_t len) const {
    int res = static_cast<int>(uchar_arr[0]);
    std::size_t index;
    for (index = 1; index < len; ++index) {
      res = (res << 8) + uchar_arr[index];
    }
    return res;
  }
  void read_img_header(std::ifstream& ifs, ImgHeader& img_header) const {
    ifs.read((char*)(&(img_header.magic_number)), 
             sizeof(img_header.magic_number));
    ifs.read((char*)(&(img_header.num_imgs)), sizeof(img_header.num_imgs));
    ifs.read((char*)(&(img_header.num_rows)), sizeof(img_header.num_rows));
    ifs.read((char*)(&(img_header.num_cols)), sizeof(img_header.num_cols));
  }
  void read_lbl_header(std::ifstream& ifs, LblHeader& lbl_header) const {
    ifs.read((char*)(&(lbl_header.magic_number)), 
             sizeof(lbl_header.magic_number));
    ifs.read((char*)(&(lbl_header.num_lbls)), sizeof(lbl_header.num_lbls));
  }
  void load_img(std::ifstream& ifs, 
                std::vector<std::vector<double>>& img_data) const {
    ImgHeader img_header;
    read_img_header(ifs, img_header);
    int magic_number = uchar_to_int(img_header.magic_number, 4);
    if (magic_number != IMG_MAGIC_NUMBER) {
      throw "Invalid MNIST file";
    }
    int num_imgs = uchar_to_int(img_header.num_imgs, 4);
    int num_rows = uchar_to_int(img_header.num_rows, 4);
    int num_cols = uchar_to_int(img_header.num_cols, 4);
    img_data.resize(num_imgs);
    unsigned char cache_read;
    for (int idx_img = 0; idx_img < num_imgs; ++idx_img) {
      img_data[idx_img].resize(num_rows * num_cols);
      for (int idx_pix = 0; idx_pix < num_rows * num_cols; ++idx_pix) {
        ifs.read((char*)(&cache_read), sizeof(cache_read));
        img_data[idx_img][idx_pix] = static_cast<double>(cache_read);
      }
    }
  }
  void load_lbl(std::ifstream& ifs, 
                std::vector<std::vector<double>>& lbl_data) const {
    LblHeader lbl_header;
    read_lbl_header(ifs, lbl_header);
    int magic_number = uchar_to_int(lbl_header.magic_number, 4);
    if (magic_number != LBL_MAGIC_NUMBER) {
      throw "Invalid MNIST file";
    }
    int num_lbls = uchar_to_int(lbl_header.num_lbls, 4);
    lbl_data.resize(num_lbls);
    unsigned char cache_read;
    for (int idx_lbl = 0; idx_lbl < num_lbls; ++idx_lbl) {
      lbl_data[idx_lbl].resize(10);
      ifs.read((char*)(&cache_read), sizeof(cache_read));
      lbl_data[idx_lbl][static_cast<int>(cache_read)] = 1;
    }
  }
  void load_img_lbl(tensor::Matrix<double>& img_data,
                    tensor::Matrix<double>& lbl_data,
                    const std::string& img_name, 
                    const std::string& lbl_name) const {
    std::ifstream ifs_img(path_ + img_name, std::ios::binary);
    std::ifstream ifs_lbl(path_ + lbl_name, std::ios::binary);
    if (ifs_img.is_open() && ifs_lbl.is_open()) {
      std::vector<std::vector<double>> stdvec_img, stdvec_lbl;
      load_img(ifs_img, stdvec_img);
      load_lbl(ifs_lbl, stdvec_lbl);
      img_data = tensor::Matrix<double>(stdvec_img);
      lbl_data = tensor::Matrix<double>(stdvec_lbl);
    } else {
      throw "Cannot open " + img_name + ", " + lbl_name + ".";
    }    
  }
};

}  // namespace data
}  // namespace nn

#endif  // NN_DATA_MINST_H_