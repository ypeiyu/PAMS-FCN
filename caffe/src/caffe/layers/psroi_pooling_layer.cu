// --------------------------------------------------------
// R-FCN
// Written by Yi Li, 2016.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/psroi_pooling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    const int divide_type,
    Dtype* top_data,
    int* mapping_channel) {

    if (divide_type == 0)
    {
          CUDA_KERNEL_LOOP(index, nthreads) {

          //****int divide_num = xx;  group_size;
          int index_mod = index % pooled_width;
          int order;

          int part_arr[] = {0,9,29,59,99};   //9 20 30 40 60

          if(index_mod >= 0 && index_mod < 9){
                order = 0;

          }else if(index_mod >= 9 && index_mod < 29){
                order = 1;

          }else if(index_mod >= 29 && index_mod < 59){
                order = 2;

          }else if(index_mod >= 59 && index_mod < 99){
                order = 3;

          }else if(index_mod >= 99 && index_mod < 159){
                order = 4;

          }


          // The output is in order (n, ctop, ph, pw)

          int ctop = (index / group_size) % output_dim;  //group_size == divide_num

          // ***************************
          int n = index / pooled_width / pooled_height / output_dim;

          //*** ROI ***
          // [start, end) interval for spatial sampling
          bottom_rois += n * 5;
          int roi_batch_ind = bottom_rois[0];
          Dtype roi_start_w =
            static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
          Dtype roi_start_h =
            static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
          Dtype roi_end_w =
            static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
          Dtype roi_end_h =
            static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

          // Force too small ROIs to be 1x1
          Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
          Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

          Dtype hstart;
          Dtype hend;
          Dtype wstart;
          Dtype wend;

          Dtype bin_size_h;
          Dtype bin_size_w;
          Dtype bin_size_h0;
          Dtype bin_size_w0;

          int divide_num_col;  // col_num
          int divide_num_row;

          switch(order){

            case 0 :
            // head
            bin_size_h = roi_height * static_cast<Dtype>(0.23);
            bin_size_w = roi_width * static_cast<Dtype>(0.7);
            bin_size_w0 = roi_width * static_cast<Dtype>(0.15);

            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w + bin_size_w0;
            wend   = roi_start_w + bin_size_w0 + bin_size_w;

            /////////
            divide_num_col = 3;  // col_num
            divide_num_row = 3;
            break;

            case 1 :
            // top-30
            bin_size_h = roi_height * static_cast<Dtype>(0.3);
            bin_size_w = roi_width;

            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;

            /////////
            divide_num_col = 5;  // col_num
            divide_num_row = 4;
            break;

            case 2 :
            // top-50
            bin_size_h = roi_height * static_cast<Dtype>(0.5);
            bin_size_w = roi_width;

            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;

             /////////
            divide_num_col = 5;  // col_num
            divide_num_row = 6;
            break;

            case 3 :
            // top-70
            bin_size_h = roi_height * static_cast<Dtype>(0.7);
            bin_size_w = roi_width;

            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;

            /////////
            divide_num_col = 5;  // col_num
            divide_num_row = 8;
            break;

            case 4 :
            // full
            bin_size_h = roi_height;
            bin_size_w = roi_width;

            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;

            /////////
            divide_num_col = 5;  // col_num
            divide_num_row = 12;
            break;
          }

          //int silce_gird = divide_num_col * divide_num_row;

          //int inner_order = (index - order * silce_gird) % silce_gird;
          int inner_order = index_mod - part_arr[order];

          int hstart_1;
          int hend_1;
          int wstart_1;
          int wend_1;


          Dtype bin_h = static_cast<Dtype>(hend-hstart) / static_cast<Dtype>(divide_num_row+0.0);
          Dtype bin_w = static_cast<Dtype>(wend-wstart) / static_cast<Dtype>(divide_num_col+0.0);

          int h_order = inner_order / divide_num_col;
          int w_order = inner_order % divide_num_col;

          hstart_1 = floor(static_cast<Dtype>(hstart + h_order * bin_h));
          hend_1   =  ceil(static_cast<Dtype>(hstart + (h_order+1) * bin_h));

          wstart_1 = floor(static_cast<Dtype>(wstart + w_order * bin_w));
          wend_1   =  ceil(static_cast<Dtype>(wstart + (w_order+1) * bin_w));

          hstart_1 = min(max(hstart_1, 0), height);
          hend_1   = min(max(hend_1,   0), height);
          wstart_1 = min(max(wstart_1, 0), width);
          wend_1   = min(max(wend_1,   0), width);

          int c = ctop*group_size + index_mod;

          hstart_1 = min(max(hstart_1, 0), height);
          hend_1   = min(max(hend_1,   0), height);
          wstart_1 = min(max(wstart_1, 0), width);
          wend_1   = min(max(wend_1,   0), width);
          bool is_empty = (hend_1 <= hstart_1) || (wend_1 <= wstart_1);

          bottom_data += (roi_batch_ind * channels + c) * height * width;
          Dtype out_sum = 0;
          for (int h = hstart_1; h < hend_1; ++h) {
            for (int w = wstart_1; w < wend_1; ++w) {
              int bottom_index = h*width + w;
              out_sum += bottom_data[bottom_index];
            }
          }

          Dtype bin_area = (hend_1 - hstart_1)*(wend_1 - wstart_1);
          top_data[index] = is_empty? 0. : out_sum/bin_area;
          mapping_channel[index] = c;
        }
    }
    else if(divide_type == 1)
    {
      CUDA_KERNEL_LOOP(index, nthreads) {

        //****int divide_num = xx;  group_size;
        int index_mod = index % pooled_width;
        int order;

        int part_arr[] = {0,6,18,38,66};   //6 12 20 28 40

        if(index_mod >= 0 && index_mod < 6){
              order = 0;

        }else if(index_mod >= 6 && index_mod < 18){
              order = 1;

        }else if(index_mod >= 18 && index_mod < 38){
              order = 2;

        }else if(index_mod >= 38 && index_mod < 66){
              order = 3;

        }else if(index_mod >= 66 && index_mod < 106){
              order = 4;

        }


        // The output is in order (n, ctop, ph, pw)

        int ctop = (index / group_size) % output_dim;  //group_size == divide_num

        // ***************************
        int n = index / pooled_width / pooled_height / output_dim;

        //*** ROI ***
        // [start, end) interval for spatial sampling
        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        Dtype roi_start_w =
          static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
        Dtype roi_start_h =
          static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
        Dtype roi_end_w =
          static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
        Dtype roi_end_h =
          static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
        Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

        Dtype hstart;
        Dtype hend;
        Dtype wstart;
        Dtype wend;

        Dtype bin_size_h;
        Dtype bin_size_w;
        Dtype bin_size_h0;
        Dtype bin_size_w0;

        int divide_num_col;  // col_num
        int divide_num_row;

        switch(order){

          case 0 :
          // head
          bin_size_h = roi_height * static_cast<Dtype>(0.23);
          bin_size_w = roi_width * static_cast<Dtype>(0.7);
          bin_size_w0 = roi_width * static_cast<Dtype>(0.15);

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w + bin_size_w0;
          wend   = roi_start_w + bin_size_w0 + bin_size_w;

          /////////
          divide_num_col = 3;  // col_num
          divide_num_row = 2;
          break;

          case 1 :
          // top-30
          bin_size_h = roi_height * static_cast<Dtype>(0.3);
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

          /////////
          divide_num_col = 4;  // col_num
          divide_num_row = 3;
          break;

          case 2 :
          // top-50
          bin_size_h = roi_height * static_cast<Dtype>(0.5);
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

            /////////
          divide_num_col = 4;  // col_num
          divide_num_row = 5;
          break;

          case 3 :
          // top-70
          bin_size_h = roi_height * static_cast<Dtype>(0.7);
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

          /////////
          divide_num_col = 4;  // col_num
          divide_num_row = 7;
          break;

          case 4 :
          // full
          bin_size_h = roi_height;
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

          /////////
          divide_num_col = 4;  // col_num
          divide_num_row = 10;
          break;
        }


        //int silce_gird = divide_num_col * divide_num_row;

        //int inner_order = (index - order * silce_gird) % silce_gird;
        int inner_order = index_mod - part_arr[order];

        int hstart_1;
        int hend_1;
        int wstart_1;
        int wend_1;


        Dtype bin_h = static_cast<Dtype>(hend-hstart) / static_cast<Dtype>(divide_num_row+0.0);
        Dtype bin_w = static_cast<Dtype>(wend-wstart) / static_cast<Dtype>(divide_num_col+0.0);

        int h_order = inner_order / divide_num_col;
        int w_order = inner_order % divide_num_col;

        hstart_1 = floor(static_cast<Dtype>(hstart + h_order * bin_h));
        hend_1   =  ceil(static_cast<Dtype>(hstart + (h_order+1) * bin_h));

        wstart_1 = floor(static_cast<Dtype>(wstart + w_order * bin_w));
        wend_1   =  ceil(static_cast<Dtype>(wstart + (w_order+1) * bin_w));

        hstart_1 = min(max(hstart_1, 0), height);
        hend_1   = min(max(hend_1,   0), height);
        wstart_1 = min(max(wstart_1, 0), width);
        wend_1   = min(max(wend_1,   0), width);

        int c = ctop*group_size + index_mod;

        hstart_1 = min(max(hstart_1, 0), height);
        hend_1   = min(max(hend_1,   0), height);
        wstart_1 = min(max(wstart_1, 0), width);
        wend_1   = min(max(wend_1,   0), width);
        bool is_empty = (hend_1 <= hstart_1) || (wend_1 <= wstart_1);

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        Dtype out_sum = 0;
        for (int h = hstart_1; h < hend_1; ++h) {
          for (int w = wstart_1; w < wend_1; ++w) {
            int bottom_index = h*width + w;
            out_sum += bottom_data[bottom_index];
          }
        }

        Dtype bin_area = (hend_1 - hstart_1)*(wend_1 - wstart_1);
        top_data[index] = is_empty? 0. : out_sum/bin_area;
        mapping_channel[index] = c;
      }
    }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, output_dim_, group_size_,divide_type_,
      top_data, mapping_channel_ptr);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PSROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    const int divide_type,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {

    if(divide_type == 0)
    {
        CUDA_KERNEL_LOOP(index, nthreads) {
        int index_mod = index % pooled_width;
        int order;

        int part_arr[] = {0,9,29,59,99};   //9 20 30 40 60

        if(index_mod >= 0 && index_mod < 9){
              order = 0;

        }else if(index_mod >= 9 && index_mod < 29){
              order = 1;

        }else if(index_mod >= 29 && index_mod < 59){
              order = 2;

        }else if(index_mod >= 59 && index_mod < 99){
              order = 3;

        }else if(index_mod >= 99 && index_mod < 159){
              order = 4;

        }

        //int n = index / pooled_width / pooled_height / output_dim;
        int n = index / pooled_width / output_dim;

        // [start, end) interval for spatial sampling
        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        Dtype roi_start_w =
          static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
        Dtype roi_start_h =
          static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
        Dtype roi_end_w =
          static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
        Dtype roi_end_h =
          static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
        Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

        Dtype hstart;
        Dtype hend;
        Dtype wstart;
        Dtype wend;

        Dtype bin_size_h;
        Dtype bin_size_w;
        Dtype bin_size_h0;
        Dtype bin_size_w0;

        int divide_num_col;  // col_num
        int divide_num_row;


        switch(order){

          case 0 :
          // head
          bin_size_h = roi_height * static_cast<Dtype>(0.23);
          bin_size_w = roi_width * static_cast<Dtype>(0.7);
          bin_size_w0 = roi_width * static_cast<Dtype>(0.15);

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w + bin_size_w0;
          wend   = roi_start_w + bin_size_w0 + bin_size_w;

          /////////
          divide_num_col = 3;  // col_num
          divide_num_row = 3;
          break;

          case 1 :
          // top-30
          bin_size_h = roi_height * static_cast<Dtype>(0.3);
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

          /////////
          divide_num_col = 5;  // col_num
          divide_num_row = 4;
          break;

          case 2 :
          // top-50
          bin_size_h = roi_height * static_cast<Dtype>(0.5);
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

           /////////
          divide_num_col = 5;  // col_num
          divide_num_row = 6;
          break;

          case 3 :
          // top-70
          bin_size_h = roi_height * static_cast<Dtype>(0.7);
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

          /////////
          divide_num_col = 5;  // col_num
          divide_num_row = 8;
          break;

          case 4 :
          // full
          bin_size_h = roi_height;
          bin_size_w = roi_width;

          hstart = roi_start_h;
          hend   = roi_start_h + bin_size_h;
          wstart = roi_start_w;
          wend   = roi_start_w + bin_size_w;

          /////////
          divide_num_col = 5;  // col_num
          divide_num_row = 12;
          break;
        }

        //int silce_gird = divide_num_col * divide_num_row;

        //int inner_order = (index - order * silce_gird) % silce_gird;
        int inner_order = index_mod - part_arr[order];
        /////////

        int hstart_1;
        int hend_1;
        int wstart_1;
        int wend_1;


        Dtype bin_h = static_cast<Dtype>(hend-hstart) / static_cast<Dtype>(divide_num_row+0.0);
        Dtype bin_w = static_cast<Dtype>(wend-wstart) / static_cast<Dtype>(divide_num_col+0.0);

        int h_order = inner_order / divide_num_col;
        int w_order = inner_order % divide_num_col;

        hstart_1 = floor(static_cast<Dtype>(hstart + h_order * bin_h));
        hend_1   =  ceil(static_cast<Dtype>(hstart + (h_order+1) * bin_h));

        wstart_1 = floor(static_cast<Dtype>(wstart + w_order * bin_w));
        wend_1   =  ceil(static_cast<Dtype>(wstart + (w_order+1) * bin_w));

        hstart_1 = min(max(hstart_1, 0), height);
        hend_1   = min(max(hend_1,   0), height);
        wstart_1 = min(max(wstart_1, 0), width);
        wend_1   = min(max(wend_1,   0), width);


        ///
        bool is_empty = (hend_1 <= hstart_1) || (wend_1 <= wstart_1);

        // Compute c at bottom
        int c = mapping_channel[index];
        Dtype* offset_bottom_diff = bottom_diff +
          (roi_batch_ind * channels + c) * height * width;
        Dtype bin_area = (hend_1 - hstart_1)*(wend_1 - wstart_1);
        Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
        for (int h = hstart_1; h < hend_1; ++h) {
          for (int w = wstart_1; w < wend_1; ++w) {
            int bottom_index = h*width + w;
            caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
          }
        }
      }
    }
    else if(divide_type==1)
    {
          CUDA_KERNEL_LOOP(index, nthreads) {
          int index_mod = index % pooled_width;
          int order;


          int part_arr[] = {0,6,18,38,66};   //6 12 20 28 40

          if(index_mod >= 0 && index_mod < 6){
                order = 0;
  
          }else if(index_mod >= 6 && index_mod < 18){
                order = 1;
  
          }else if(index_mod >= 18 && index_mod < 38){
                order = 2;
  
          }else if(index_mod >= 38 && index_mod < 66){
                order = 3;
  
          }else if(index_mod >= 66 && index_mod < 106){
                order = 4;
  
          }

          
          //int n = index / pooled_width / pooled_height / output_dim;
          int n = index / pooled_width / output_dim;

          // [start, end) interval for spatial sampling
          bottom_rois += n * 5;
          int roi_batch_ind = bottom_rois[0];
          Dtype roi_start_w =
            static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
          Dtype roi_start_h =
            static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
          Dtype roi_end_w =
            static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
          Dtype roi_end_h =
            static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

          // Force too small ROIs to be 1x1
          Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
          Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

          Dtype hstart;
          Dtype hend;
          Dtype wstart;
          Dtype wend;

          Dtype bin_size_h;
          Dtype bin_size_w;
          Dtype bin_size_h0;
          Dtype bin_size_w0;

          int divide_num_col;  // col_num
          int divide_num_row;

          switch(order){

            case 0 :
            // head
            bin_size_h = roi_height * static_cast<Dtype>(0.23);
            bin_size_w = roi_width * static_cast<Dtype>(0.7);
            bin_size_w0 = roi_width * static_cast<Dtype>(0.15);
  
            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w + bin_size_w0;
            wend   = roi_start_w + bin_size_w0 + bin_size_w;
  
            /////////
            divide_num_col = 3;  // col_num
            divide_num_row = 2;
            break;
  
            case 1 :
            // top-30
            bin_size_h = roi_height * static_cast<Dtype>(0.3);
            bin_size_w = roi_width;
  
            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;
  
            /////////
            divide_num_col = 4;  // col_num
            divide_num_row = 3;
            break;
  
            case 2 :
            // top-50
            bin_size_h = roi_height * static_cast<Dtype>(0.5);
            bin_size_w = roi_width;
  
            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;
  
              /////////
            divide_num_col = 4;  // col_num
            divide_num_row = 5;
            break;
  
            case 3 :
            // top-70
            bin_size_h = roi_height * static_cast<Dtype>(0.7);
            bin_size_w = roi_width;
  
            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;
  
            /////////
            divide_num_col = 4;  // col_num
            divide_num_row = 7;
            break;
  
            case 4 :
            // full
            bin_size_h = roi_height;
            bin_size_w = roi_width;
  
            hstart = roi_start_h;
            hend   = roi_start_h + bin_size_h;
            wstart = roi_start_w;
            wend   = roi_start_w + bin_size_w;
  
            /////////
            divide_num_col = 4;  // col_num
            divide_num_row = 10;
            break;
          }

          

          //int silce_gird = divide_num_col * divide_num_row;

          //int inner_order = (index - order * silce_gird) % silce_gird;
          int inner_order = index_mod - part_arr[order];
          /////////

          int hstart_1;
          int hend_1;
          int wstart_1;
          int wend_1;


          Dtype bin_h = static_cast<Dtype>(hend-hstart) / static_cast<Dtype>(divide_num_row+0.0);
          Dtype bin_w = static_cast<Dtype>(wend-wstart) / static_cast<Dtype>(divide_num_col+0.0);

          int h_order = inner_order / divide_num_col;
          int w_order = inner_order % divide_num_col;

          hstart_1 = floor(static_cast<Dtype>(hstart + h_order * bin_h));
          hend_1   =  ceil(static_cast<Dtype>(hstart + (h_order+1) * bin_h));

          wstart_1 = floor(static_cast<Dtype>(wstart + w_order * bin_w));
          wend_1   =  ceil(static_cast<Dtype>(wstart + (w_order+1) * bin_w));

          hstart_1 = min(max(hstart_1, 0), height);
          hend_1   = min(max(hend_1,   0), height);
          wstart_1 = min(max(wstart_1, 0), width);
          wend_1   = min(max(wend_1,   0), width);


          ///
          bool is_empty = (hend_1 <= hstart_1) || (wend_1 <= wstart_1);

          // Compute c at bottom
          int c = mapping_channel[index];
          Dtype* offset_bottom_diff = bottom_diff +
            (roi_batch_ind * channels + c) * height * width;
          Dtype bin_area = (hend_1 - hstart_1)*(wend_1 - wstart_1);
          Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
          for (int h = hstart_1; h < hend_1; ++h) {
            for (int w = wstart_1; w < wend_1; ++w) {
              int bottom_index = h*width + w;
              caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
            }
          }
        }
    }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
      top[0]->num(), spatial_scale_, channels_, height_, width_,
      pooled_height_, pooled_width_, output_dim_,divide_type_, bottom_diff,
      bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PSROIPoolingLayer);

}  // namespace caffe
