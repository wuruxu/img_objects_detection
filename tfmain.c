#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>
#include <signal.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <turbojpeg.h>
#include <cuda.h>
#include <tensorflow/c/c_api.h>
#include <nppi.h>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include <microhttpd.h>
#include <pthread.h>
#include <sqlite3.h>
#include <openssl/md5.h>
#include <cairo.h>
#include <fts.h>
#include <cairo_jpg.h>
#include <libxml/tree.h>

#include "yuv_rgb.h"

#define DEBUG(...) fprintf(stderr, __VA_ARGS__)

#define UC_FPS 3
#define SCORES_PASS 0.65f

const char* USB_CAMERA_NULL = "/dev/null";
const char* USB_CAMERA_DEV0 = "/dev/video0";
const char* USB_CAMERA_DEV1 = "/dev/video1";
const char* TF_OUTPUT_DB = "/tmp/tfoutput.db";

struct yuv_buffer {
  void *start;
  size_t length;
};
//const char* classes_name[] = {"dummy", "earsoup", "rice", "lacery", "octagon", "triangle", "small_shellw", "rect"};
//const char* classes_name[] = {"dummy", "lacery", "triangle", "small_shellw" };
//const char* classes_name[] = {"dummy", "earsoup", "lacery", "octagon", "triangle", "rect", "rice", "egg"};
const char* classes_name[] = {"dummy", "汤碗", "花边", "八边", "三角", "方盘", "米饭", "绿汤"};
#define CLASSES_NAME_SIZE sizeof(classes_name)/sizeof(const char*)

static int g_GET_reqcnt = 0;
const char* tensor_name[] = {"boxes", "scores", "classes", "count"};
typedef struct {
  const char* tfHandle;
  TF_Session* tfSession;
  TF_Status* tfStatus;
  TF_Graph* tfGraph;

  TF_Operation* inop;
  TF_Operation *oo_scores, *oo_boxes, *oo_classes, *oo_count;

  //struct event* int_sigev;
  //struct event* ucev;
  //struct event_base* base;
  struct yuv_buffer* buffers;
  int n_buffers;
  int vfd;
  int img_width, img_height;
  unsigned char* yuvimg;
  void* yuvimg_gpu;
  struct MHD_Daemon* httpd;
  pthread_mutex_t mutex;
  char output_json[8192];
  char output_csv[8192];
  sqlite3* db;
  void* rgbbuf;
  int rgbsize;
  cairo_surface_t* surface;

  char* autolabel_imgdir;
  char* pbfn;
  char* camera;
} gotensor_t;

void free_my_buffer(void* data, size_t length) {                                             
  free(data);                                                                       
}

static int xioctl(int fd, int request, void* argp) {
  int r;

  do r = v4l2_ioctl(fd, request, argp);
  while (-1 == r && EINTR == errno);

  return r;
}

TF_Buffer* load_graph_file(const char* pbfile) {
  int nsize = 0;
  void* pbdata = NULL;
  TF_Buffer* buf = NULL;
  FILE* fp = fopen(pbfile, "rb");

  if(fp != NULL) {
    fseek(fp, 0, SEEK_END);
    nsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    pbdata = malloc(nsize);
    fread(pbdata, nsize, 1, fp);
    fclose(fp);
    DEBUG("graph_file %s nsize = %d\n", pbfile, nsize);

    buf = TF_NewBuffer();

    buf->data = pbdata;
    buf->length = nsize;
    buf->data_deallocator = free_my_buffer;
  }

  return buf;
}

static void free_cuda_buffer(void* data, size_t length) {
  cudaFreeHost(data);
}

TF_Buffer* load_graph_file_gpu(const char* pbfile) {
  int nsize = 0;
  void* pbdata = NULL;
  TF_Buffer* buf = NULL;
  FILE* fp = fopen(pbfile, "rb");

  if(fp != NULL) {
    fseek(fp, 0, SEEK_END);
    nsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if(cudaSuccess != cudaHostAlloc(&pbdata, nsize, cudaHostAllocMapped)) {
      DEBUG("cudaHostAlloc failed \n");
    }
    fread(pbdata, nsize, 1, fp);
    fclose(fp);
    DEBUG("graph_file %s nsize = %d\n", pbfile, nsize);

    buf = TF_NewBuffer();

    buf->data = pbdata;
    buf->length = nsize;
    buf->data_deallocator = free_cuda_buffer;
  }

  return buf;
}

void free_yuv_buffer(void* data, size_t nsize, void* arg) {
  tjFree(data);
}

TF_Tensor* create_image_tensor_from_v4l2(const char* jpegfn, float* img_width, float* img_height) {
  int nbytes = 0;
  int64_t dims[] = {1, 0, 0, 3};
  //unsigned char* nparray[] = {NULL};
  FILE* fp = fopen(jpegfn, "rb");
  tjhandle tjInstance = tjInitDecompress();
  unsigned char* jpegBuf = NULL, *imgBuf = NULL;
  int jpegSize = 0, width = 0, height = 0;
  int inSubsamp = 0, inColorspace = 0;

  if(fp != NULL) {
    fseek(fp, 0, SEEK_END);
    jpegSize = ftell(fp);
    if(jpegSize > 0) {
      if ((jpegBuf = (unsigned char *)tjAlloc(jpegSize)) == NULL) {
      } else {
        fseek(fp, 0, SEEK_SET);
        fread(jpegBuf, jpegSize, 1, fp);
        fclose(fp);
      }
    }
  }

  if(jpegBuf != NULL) {
    if(tjDecompressHeader3(tjInstance, jpegBuf, jpegSize, &width, &height, &inSubsamp, &inColorspace) < 0) {
      DEBUG("%s isn't valid jpeg file, error = %s\n", jpegfn, tjGetErrorStr2(tjInstance));
    }
    DEBUG("%s size = %d, width = %d, height = %d, tjInstance = %p\n", jpegfn, jpegSize, width, height, tjInstance);
    *img_width = (float) width, *img_height = (float) height;
    nbytes = width * height * tjPixelSize[TJPF_RGB];
    imgBuf = (unsigned char *)tjAlloc(nbytes);
    if(tjDecompress2(tjInstance, jpegBuf, jpegSize, imgBuf, width, 0, height, TJPF_RGB, 0) < 0) {
      DEBUG("decompress %s as TJPF_RGB failed\n", jpegfn);
    }

    tjFree(jpegBuf);
    tjDestroy(tjInstance);

    dims[0] = 1, dims[1] = height, dims[2] = width, dims[3] = 3;
    //dims[0] = 3, dims[1] = width, dims[2] = height;//, dims[3] = 1;
    return TF_NewTensor(TF_UINT8, dims, 4, imgBuf, nbytes, free_yuv_buffer, imgBuf);
  }

  return NULL;
}

void free_rgbbuf (void* data, size_t nsize, void* arg) {
  //free(data);
  cudaFreeHost(data);
}

TF_Tensor* create_tensor_from_rgbbuf(unsigned char* rgbbuf, int nbytes, int height, int width) {
  int64_t dims[] = {1, 0, 0, 3};

  if(rgbbuf != NULL) {
    dims[0] = 1, dims[1] = height, dims[2] = width, dims[3] = 3;
    return TF_NewTensor(TF_UINT8, dims, 4, rgbbuf, nbytes, free_rgbbuf, rgbbuf);
  }

  return NULL;
}

void show_tf_devices(TF_Graph* graph) {
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Status* status = TF_NewStatus();
  TF_Session* session = TF_NewSession(graph, opts, status);
  TF_DeleteSessionOptions(opts);

  TF_DeviceList* devices = TF_SessionListDevices(session, status);
  if(TF_GetCode(status) != TF_OK) {
    DEBUG("show_tf_devices.TF_SessionListDevices failed\n");
  } else {
    int ndev = TF_DeviceListCount(devices);
    DEBUG("tensorflow device number %d\n", ndev);
    for(int i = 0; i < ndev; ++i) {
      DEBUG("device.%d name = %s, type = %s\n", i, 
         TF_DeviceListName(devices, i, status),
         TF_DeviceListType(devices, i, status)
      );
    }
  }

  TF_DeleteDeviceList(devices);
  TF_DeleteSession(session, status);
  TF_DeleteStatus(status);
}

int image_detection_setup(gotensor_t* ptr) {
  struct timeval tm;
  long long stm = 0, etm = 0;

  if(ptr->tfHandle != NULL) return 0;

  gettimeofday(&tm, NULL);
  stm = tm.tv_sec * 1000LL + tm.tv_usec/1000;

  if(ptr != NULL) {
    TF_Output feeds[] = {{ptr->inop, 0}};
    TF_Output fetches[] = {{ptr->oo_boxes, 0}, {ptr->oo_scores, 0}, {ptr->oo_classes, 0}, {ptr->oo_count, 0}};
    TF_SessionPRunSetup(ptr->tfSession, feeds, 1, fetches, 4, NULL, 0, &ptr->tfHandle, ptr->tfStatus);
    if(TF_GetCode(ptr->tfStatus) == TF_OK) {
      DEBUG("\nTF_SessionPRunSetup (%s), tfHandle = '%s'\n", TF_Message(ptr->tfStatus), ptr->tfHandle);
    }
  }

  gettimeofday(&tm, NULL);
  etm = tm.tv_sec * 1000LL + tm.tv_usec/1000;
  DEBUG("image_detection_setup time = %lld msec\n\n", etm-stm);

  return 0;
}

static void draw_tfoutput_for_imgbuf(gotensor_t* go, cairo_t* cr, int output_idx, int classes, float scores, float x0, float y0, float x1, float y1) {
  //cairo_surface_t* surface = NULL;
  //cairo_t *cr = NULL;
  double corner_radius = go->img_height/10.0;
  double radius = corner_radius/3.0;
  double degrees = 3.14159265358979323846/180.0;
  cairo_status_t e = 0;
  char scores_text[256] = {0};
  char classes_text[256] = {0};
  double tx, ty;
  double line_width = 3.0 * scores;
  //int nstride = cairo_format_stride_for_width (CAIRO_FORMAT_RGB24, go->img_width);

  //surface = cairo_image_surface_create_for_data(go->rgbbuf, CAIRO_FORMAT_RGB24, go->img_width, go->img_height, nstride);
  //cr = cairo_create(go->surface);

  cairo_new_sub_path (cr);
  cairo_arc (cr, x1 - radius, y0 + radius, radius, -90 * degrees, 0 * degrees);
  cairo_arc (cr, x1 - radius, y1 - radius, radius, 0 * degrees, 90 * degrees);
  cairo_arc (cr, x0 + radius, y1 - radius, radius, 90 * degrees, 180 * degrees);
  cairo_arc (cr, x0 + radius, y0 + radius, radius, 180 * degrees, 270 * degrees);
  cairo_close_path (cr);
  cairo_set_source_rgba (cr, 0.6, 0, 0, scores + 0.2);
  cairo_set_line_width(cr, line_width);

  cairo_stroke (cr);

  //cairo_set_source_rgba (cr, 0, 0, 0, 1.0);
  cairo_set_source_rgba (cr, 1, 1, 1, scores + 0.25);
  cairo_text_extents_t extents;
  //cairo_select_font_face(cr, "Serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
  //cairo_select_font_face(cr, "Noto Mono", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  //cairo_select_font_face(cr, "Droid Sans Fallback", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  cairo_select_font_face(cr, "WenQuanYi Zen Hei", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);
  //cairo_select_font_face(cr, "Noto Mono", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
  cairo_set_font_size (cr, 32.0);
  if(classes >= 0 && classes < CLASSES_NAME_SIZE)
    snprintf(scores_text, 256, "%s %.2f",classes_name[classes], scores);
  else
    snprintf(scores_text, 256, "***%d*** %.2f", classes, scores);
  cairo_text_extents (cr, scores_text, &extents);
  //DEBUG("extents(%f, %f, %f, %f)\n", extents.width, extents.height, extents.x_bearing, extents.y_bearing);
  tx = (x0 + x1)/2 - (extents.width/2 + extents.x_bearing); //x0 + (x1-x0)/2 - (extents.width/2 + extents.x_bearing);
  //tx = x0 + line_width - (extents.width/2 + extents.x_bearing); //x0 + (x1-x0)/2 - (extents.width/2 + extents.x_bearing);
  //ty = y1 + (extents.height) + line_width;
  ty = (output_idx * 10) + (y0 + y1)/2 - (extents.height/2 + extents.y_bearing); //y1 + (extents.height + extents.y_bearing) - line_width;
  cairo_move_to (cr, tx, ty);
  cairo_show_text(cr, scores_text);

#if 0
  //cairo_set_source_rgba (cr, 0, 0, 0, 1.0);
  if(classes >= 0 && classes < CLASSES_NAME_SIZE)
    snprintf(classes_text, 256, "%s", classes_name[classes]);
  else
    snprintf(classes_text, 256, "***%d***", classes);
  cairo_text_extents (cr, classes_text, &extents);
  //DEBUG("extents(%f, %f, %f, %f)\n", extents.width, extents.height, extents.x_bearing, extents.y_bearing);
  tx = x0 + (x1-x0)/2 - (extents.width/2 + extents.x_bearing);
  ty = y0 + (extents.height/2 - extents.y_bearing) + line_width;
  //ty = y0 + (extents.height) + line_width;
  cairo_move_to (cr, tx, ty);
  cairo_show_text(cr, classes_text);
#endif

  //e = cairo_image_surface_write_to_jpeg(surface, "/tmp/output.jpg", 90);
  //if(CAIRO_STATUS_SUCCESS != e) {
  //  DEBUG("cairo_image_surface_write_to_jpeg failed, e = %x\n", e);
  //}

  //cairo_destroy(cr);
  //cairo_surface_destroy(surface);
}

void do_rgbimg_detection(gotensor_t* ptr, unsigned char* rgbbuf, int nbytes, int img_width, int img_height) {
  //struct timeval tm;
  //long long stm = 0, etm = 0;
  float width = (float)img_width , height = (float)img_height;

  //gettimeofday(&tm, NULL);
  //stm = tm.tv_sec * 1000LL + tm.tv_usec/1000;

  if(ptr != NULL) {
    TF_Status* status = TF_NewStatus();
    TF_Output feeds[] = {{ptr->inop, 0}};
    TF_Output fetches[] = {{ptr->oo_boxes, 0}, {ptr->oo_scores, 0}, {ptr->oo_classes, 0}, {ptr->oo_count, 0}};

    TF_Tensor* v4l2_tensor = create_tensor_from_rgbbuf(rgbbuf, nbytes, img_height, img_width);
    TF_Tensor* in_tensors[] = {v4l2_tensor};
    TF_Tensor* oo_tensor[] = {NULL, NULL, NULL, NULL};

    DEBUG("do_rgbimg_detection, v4l2_tensor = %p, tfSession = %p, tfHandle = '%s'\n", v4l2_tensor, ptr->tfSession, ptr->tfHandle, ptr->tfHandle);
    if(v4l2_tensor != NULL) {
      int ret = 0;
      int count = 0;
      float *boxes_data, *scores_data, *classes_data, *count_data;
      TF_SessionPRun(ptr->tfSession, ptr->tfHandle, feeds, in_tensors, 1, fetches, oo_tensor, 4, NULL, 0, status);
      ret = TF_GetCode(status) != TF_OK;
      DEBUG("do_rgbimg_detection.TF_SessionPRun status %c= TF_OK (%s), tm = %ld\n", ret ? '!' : '=', TF_Message(status), time(NULL));

      boxes_data = TF_TensorData(oo_tensor[0]), scores_data = TF_TensorData(oo_tensor[1]), classes_data = TF_TensorData(oo_tensor[2]), count_data = TF_TensorData(oo_tensor[3]);
      count = (int) *(float *)count_data;
      DEBUG("oo_tensor.data (%p, %p, %p, %p), num_detections = %d\n", boxes_data, scores_data, classes_data, count_data, count);

      //for(int i = 0; i < 4; i++) {
      //  TF_Tensor* t = oo_tensor[i];
      //  int nbytes = TF_TensorByteSize(t);
      //  switch(TF_NumDims(oo_tensor[i])) {
      //  case 1:
      //  DEBUG("oo_tensor[%d].name = %s (%d)", i, tensor_name[i], TF_Dim(t, 0));
      //  break;
      //  case 2:
      //  DEBUG("oo_tensor[%d].name = %s (%d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1));
      //  break;
      //  case 3:
      //  DEBUG("oo_tensor[%d].name = %s (%d, %d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1), TF_Dim(t, 2));
      //  break;
      //  case 4:
      //  DEBUG("oo_tensor[%d].name = %s (%d, %d, %d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1), TF_Dim(t, 2), TF_Dim(t, 3));
      //  break;
      //  }
      //  DEBUG(" type = %x, bytesize = %d\n", TF_TensorType(t), nbytes);
      //}


      cairo_t* cr = cairo_create(ptr->surface);
      char* oojson = ptr->output_json;
      char* oocsv = ptr->output_csv;
      oojson += sprintf(oojson, "{\"Recognition\":[");
      for(int i = 0; i < count; i++, scores_data++, classes_data++) {
        float scores;
        float x0, y0, x1, y1;
        int classes;
        scores = *(float *) scores_data;
        classes = (int) *(float *) classes_data;
        y0 = *(float *) boxes_data, x0 = *(float *) (boxes_data + 1), y1 = *(float *) (boxes_data + 2), x1 = *(float *) (boxes_data + 3);
        //DEBUG("raw boxes(%.5f, %5f, %5f, %5f)\n", x0, y0, x1, y1);
        x0 *= width, x1 *= width;
        y0 *= height, y1 *= height;

        if(scores > SCORES_PASS) {
          const char* classes_name_str = "******";
          DEBUG("oo_tensor[%d], scores = %.2f, classes %d, (%.2f, %.2f, %.2f, %.2f)\n", i, scores, classes, x0, y0, x1, y1);
          oocsv += sprintf(oocsv, "%d,%.3f,%.3f,%.3f,%.3f,%.3f;", classes, scores, x0, y0, x1, y1);
          if(classes >= 0 && classes < CLASSES_NAME_SIZE) {
            classes_name_str = classes_name[classes];
          }
          if(i > 0) {
            oojson += sprintf(oojson, ",{\"outputName\":\"%s\",\"outputClasses\":%d,\"outputScores\":%.2f,\"x0\":%.2f,\"y0\":%.2f,\"x1\":%.2f,\"y1\":%.2f}", classes_name_str, classes, scores, x0, y0, x1, y1);
          } else {
            oojson += sprintf(oojson, "{\"outputName\":\"%s\",\"outputClasses\":%d,\"outputScores\":%.2f,\"x0\":%.2f,\"y0\":%.2f,\"x1\":%.2f,\"y1\":%.2f}", classes_name_str, classes, scores, x0, y0, x1, y1);
          }
          draw_tfoutput_for_imgbuf(ptr, cr, i, classes, scores, x0, y0, x1, y1);
        }
        boxes_data += 4;
      }
      sprintf(oojson, "]}");
      //draw_tfoutput_for_imgbuf(ptr, cr, 0, 2, 0.9, 20.0, 20.0, 360.0, 360.0);
      //draw_tfoutput_for_imgbuf(ptr, cr, 1, 3, 0.5, 30.0, 420.0, 390.0, 700.0);

      cairo_destroy(cr);

      TF_DeleteTensor(oo_tensor[0]);
      TF_DeleteTensor(oo_tensor[1]);
      TF_DeleteTensor(oo_tensor[2]);
      TF_DeleteTensor(oo_tensor[3]);
    }
    TF_DeletePRunHandle(ptr->tfHandle);
    TF_DeleteStatus(status);
    ptr->tfHandle = NULL;
  }
  //gettimeofday(&tm, NULL);
  //etm = tm.tv_sec * 1000LL + tm.tv_usec/1000;
  //DEBUG("do_rgbimg_detection time = %lld msec\n\n", etm-stm);
  //DEBUG("output=%s\n", ptr->output_json);
}

typedef void (* tf_output_callback_t) (void* data, float scores, int classes, float x0, float y0, float x1, float y1) ;

void do_jpegimg_detection_ex(gotensor_t* ptr, const char* jpegfn, tf_output_callback_t outputcb, void* data) {
  struct timeval tm;
  long long stm = 0, etm = 0;
  float width, height;

  gettimeofday(&tm, NULL);
  stm = tm.tv_sec * 1000LL + tm.tv_usec/1000;

  if(ptr != NULL) {
    TF_Status* status = TF_NewStatus();
    TF_Output feeds[] = {{ptr->inop, 0}};
    TF_Output fetches[] = {{ptr->oo_boxes, 0}, {ptr->oo_scores, 0}, {ptr->oo_classes, 0}, {ptr->oo_count, 0}};

    TF_Tensor* v4l2_tensor = create_image_tensor_from_v4l2(jpegfn, &width, &height);
    TF_Tensor* in_tensors[] = {v4l2_tensor};
    TF_Tensor* oo_tensor[] = {NULL, NULL, NULL, NULL};

    DEBUG("do_jpegimg_detection_ex, v4l2_tensor = %p, tfSession = %p, tfHandle = %p '%s'\n", v4l2_tensor, ptr->tfSession, ptr->tfHandle, ptr->tfHandle);
    if(v4l2_tensor != NULL) {
      int ret = 0;
      int count = 0;
      float *boxes_data, *scores_data, *classes_data, *count_data;
      TF_SessionPRun(ptr->tfSession, ptr->tfHandle, feeds, in_tensors, 1, fetches, oo_tensor, 4, NULL, 0, status);
      ret = TF_GetCode(status) != TF_OK;
      DEBUG("do_jpegimg_detection_ex.TF_SessionPRun status %c= TF_OK (%s), tm = %ld\n", ret ? '!' : '=', TF_Message(status), time(NULL));

      boxes_data = TF_TensorData(oo_tensor[0]), scores_data = TF_TensorData(oo_tensor[1]), classes_data = TF_TensorData(oo_tensor[2]), count_data = TF_TensorData(oo_tensor[3]);
      count = (int) *(float *)count_data;
      DEBUG("oo_tensor.data (%p, %p, %p, %p), num_detections = %d\n", boxes_data, scores_data, classes_data, count_data, count);

      char* oojson = ptr->output_json;
      oojson += sprintf(oojson, "{\"Recognition\":[");
      for(int i = 0; i < count; i++, scores_data++, classes_data++) {
        float scores;
        float x0, y0, x1, y1;
        int classes;
        scores = *(float *) scores_data;
        classes = (int) *(float *) classes_data;
        y0 = *(float *) boxes_data, x0 = *(float *) (boxes_data + 1), y1 = *(float *) (boxes_data + 2), x1 = *(float *) (boxes_data + 3);
        x0 *= width, x1 *= width;
        y0 *= height, y1 *= height;

        if(scores > SCORES_PASS) {
          DEBUG("oo_tensor[%d], scores = %.2f, classes %d, (%.2f, %.2f, %.2f, %.2f)\n", i, scores, classes, x0, y0, x1, y1);
          if(outputcb != NULL) {
            outputcb(data, scores, classes, x0, y0, x1, y1);
          }
          if(i > 0) {
            oojson += sprintf(oojson, ",{\"outputClasses\":%d,\"outputScores\":%.3f,\"x0\":%.3f,\"y0\":%.3f,\"x1\":%.3f,\"y1\":%.3f}", classes, scores, x0, y0, x1, y1);
          } else {
            oojson += sprintf(oojson, "{\"outputClasses\":%d,\"outputScores\":%.3f,\"x0\":%.3f,\"y0\":%.3f,\"x1\":%.3f,\"y1\":%.3f}", classes, scores, x0, y0, x1, y1);
          }
          DEBUG("output %s\n", ptr->output_json);
        }
        boxes_data += 4;
      }
      sprintf(oojson, "]}");

      TF_DeleteTensor(oo_tensor[0]);
      TF_DeleteTensor(oo_tensor[1]);
      TF_DeleteTensor(oo_tensor[2]);
      TF_DeleteTensor(oo_tensor[3]);
    }
    TF_DeletePRunHandle(ptr->tfHandle);
    TF_DeleteStatus(status);
    ptr->tfHandle = NULL;
  }
  gettimeofday(&tm, NULL);
  etm = tm.tv_sec * 1000LL + tm.tv_usec/1000;
  DEBUG("do_jpegimg_detection_ex %s time = %lld msec\n\n", jpegfn, etm-stm);
  DEBUG("output = %s\n", ptr->output_json);
}

#if 0
void do_jpegimg_detection_ex(gotensor_t* ptr, const char* jpegfn) {
  struct timeval tm;
  long long stm = 0, etm = 0;
  float width, height;

  gettimeofday(&tm, NULL);
  stm = tm.tv_sec * 1000LL + tm.tv_usec/1000;

  if(ptr != NULL) {
    TF_Status* status = TF_NewStatus();
    TF_Output feeds[] = {{ptr->inop, 0}};
    TF_Output fetches[] = {{ptr->oo_boxes, 0}, {ptr->oo_scores, 0}, {ptr->oo_classes, 0}, {ptr->oo_count, 0}};

    TF_Tensor* v4l2_tensor = create_image_tensor_from_v4l2(jpegfn, &width, &height);
    TF_Tensor* in_tensors[] = {v4l2_tensor};
    TF_Tensor* oo_tensor[] = {NULL, NULL, NULL, NULL};

    DEBUG("do_jpegimg_detection_ex, v4l2_tensor = %p, tfSession = %p, tfHandle = %p '%s'\n", v4l2_tensor, ptr->tfSession, ptr->tfHandle, ptr->tfHandle);
    if(v4l2_tensor != NULL) {
      int ret = 0;
      int count = 0;
      float *boxes_data, *scores_data, *classes_data, *count_data;
      TF_SessionPRun(ptr->tfSession, ptr->tfHandle, feeds, in_tensors, 1, fetches, oo_tensor, 4, NULL, 0, status);
      ret = TF_GetCode(status) != TF_OK;
      DEBUG("do_jpegimg_detection_ex.TF_SessionPRun status %c= TF_OK (%s), tm = %ld\n", ret ? '!' : '=', TF_Message(status), time(NULL));

      boxes_data = TF_TensorData(oo_tensor[0]), scores_data = TF_TensorData(oo_tensor[1]), classes_data = TF_TensorData(oo_tensor[2]), count_data = TF_TensorData(oo_tensor[3]);
      count = (int) *(float *)count_data;
      DEBUG("oo_tensor.data (%p, %p, %p, %p), num_detections = %d\n", boxes_data, scores_data, classes_data, count_data, count);

      //for(int i = 0; i < 4; i++) {
      //  TF_Tensor* t = oo_tensor[i];
      //  int nbytes = TF_TensorByteSize(t);
      //  switch(TF_NumDims(oo_tensor[i])) {
      //  case 1:
      //  DEBUG("oo_tensor[%d].name = %s (%d)", i, tensor_name[i], TF_Dim(t, 0));
      //  break;
      //  case 2:
      //  DEBUG("oo_tensor[%d].name = %s (%d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1));
      //  break;
      //  case 3:
      //  DEBUG("oo_tensor[%d].name = %s (%d, %d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1), TF_Dim(t, 2));
      //  break;
      //  case 4:
      //  DEBUG("oo_tensor[%d].name = %s (%d, %d, %d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1), TF_Dim(t, 2), TF_Dim(t, 3));
      //  break;
      //  }
      //  DEBUG(" type = %x, bytesize = %d\n", TF_TensorType(t), nbytes);
      //}

      char* oojson = ptr->output_json;
      oojson += sprintf(oojson, "{\"Recognition\":[");
      for(int i = 0; i < count; i++, scores_data++, classes_data++) {
        float scores;
        float x0, y0, x1, y1;
        int classes;
        scores = *(float *) scores_data;
        classes = (int) *(float *) classes_data;
        x0 = *(float *) boxes_data, y0 = *(float *) (boxes_data + 1), x1 = *(float *) (boxes_data + 2), y1 = *(float *) (boxes_data + 3);
        x0 *= width, x1 *= width;
        y0 *= height, y1 *= height;

        if(scores > 0.3f) {
          DEBUG("oo_tensor[%d], scores = %.2f, classes %d, (%.2f, %.2f, %.2f, %.2f)\n", i, scores, classes, x0, y0, x1, y1);
          if(i > 0) {
            oojson += sprintf(oojson, ",{\"outputClasses\":%d,\"outputScores\":%.3f,\"x0\":%.3f,\"y0\":%.3f,\"x1\":%.3f,\"y1\":%.3f}", classes, scores, x0, y0, x1, y1);
          } else {
            oojson += sprintf(oojson, "{\"outputClasses\":%d,\"outputScores\":%.3f,\"x0\":%.3f,\"y0\":%.3f,\"x1\":%.3f,\"y1\":%.3f}", classes, scores, x0, y0, x1, y1);
          }
          DEBUG("output %s\n", ptr->output_json);
        }
        boxes_data += 4;
      }
      sprintf(oojson, "]}");

      TF_DeleteTensor(oo_tensor[0]);
      TF_DeleteTensor(oo_tensor[1]);
      TF_DeleteTensor(oo_tensor[2]);
      TF_DeleteTensor(oo_tensor[3]);
    }
    TF_DeletePRunHandle(ptr->tfHandle);
    TF_DeleteStatus(status);
    ptr->tfHandle = NULL;
  }
  gettimeofday(&tm, NULL);
  etm = tm.tv_sec * 1000LL + tm.tv_usec/1000;
  DEBUG("do_jpegimg_detection_ex %s time = %lld msec\n\n", jpegfn, etm-stm);
  DEBUG("output = %s\n", ptr->output_json);
}
#endif

static void _output_xml_cb(void* data, float scores, int classes, float x0, float y0, float x1, float y1) {
  xmlNodePtr root = (xmlNodePtr )data;
  xmlNodePtr obj = xmlNewChild(root, NULL, "object", NULL);
  const char* classes_name_str = "******";

  if(classes >= 0 && classes < CLASSES_NAME_SIZE) {
    classes_name_str = classes_name[classes];
  }


  xmlNewChild(obj, NULL, "name", classes_name_str);
  xmlNewChild(obj, NULL, "pose", "Unspecified");
  xmlNewChild(obj, NULL, "truncated", "0");
  xmlNewChild(obj, NULL, "difficult", "0");

  xmlNodePtr bndbox = xmlNewChild(obj, NULL, "bndbox", NULL);

  char xmin[16] = {0}, ymin[16] = {0};
  char xmax[16] = {0}, ymax[16] = {0};
  snprintf(xmin, 16, "%d", (int)x0);
  snprintf(ymin, 16, "%d", (int)y0);
  snprintf(xmax, 16, "%d", (int)x1);
  snprintf(ymax, 16, "%d", (int)y1);
  xmlNewChild(bndbox, NULL, "xmin", xmin);
  xmlNewChild(bndbox, NULL, "ymin", ymin);
  xmlNewChild(bndbox, NULL, "xmax", xmax);
  xmlNewChild(bndbox, NULL, "ymax", ymax);
}

void do_image_detection(TF_Graph* graph,TF_Session* session, const char* jpegfn) {
  TF_Status* status = TF_NewStatus();
  struct timeval tm;
  long long stm = 0, etm = 0;
  float width, height;

  gettimeofday(&tm, NULL);
  stm = tm.tv_sec * 1000LL + tm.tv_usec/1000;

  TF_Operation* inop = TF_GraphOperationByName(graph, "image_tensor");
  //DEBUG("opname = %s, optype = %s, opnum = %d, %s\n", TF_OperationName(inop), TF_OperationOpType(inop), TF_OperationNumInputs(inop), TF_OperationDevice(inop));

  TF_Operation* oo_scores = TF_GraphOperationByName(graph, "detection_scores");
  TF_Operation* oo_boxes = TF_GraphOperationByName(graph, "detection_boxes");
  TF_Operation* oo_classes = TF_GraphOperationByName(graph, "detection_classes");
  TF_Operation* oo_count = TF_GraphOperationByName(graph, "num_detections");
  //DEBUG("oo_scores = %p, oo_boxes = %p, oo_classes = %p, oo_count = %p, devicename = %s\n", oo_scores, oo_boxes, oo_classes, oo_count, TF_OperationDevice(oo_scores));

  DEBUG("do_image_detection %s\n", jpegfn);
  TF_Output feeds[] = {{inop, 0}};
  TF_Tensor* v4l2_tensor = create_image_tensor_from_v4l2(jpegfn, &width, &height);
  TF_Tensor* in_tensors[] = {NULL};
  TF_Output fetches[] = {{oo_boxes, 0}, {oo_scores, 0}, {oo_classes, 0}, {oo_count, 0}};
  TF_Tensor* oo_tensor[] = {NULL, NULL, NULL, NULL};

  DEBUG("*v4l2_tensor = %p, tm = %ld\n", v4l2_tensor, time(NULL));
  if(v4l2_tensor != NULL) {
    int ret = 0;
    int count = 0;
    in_tensors[0] = v4l2_tensor;
    float *boxes_data, *scores_data, *classes_data, *count_data;
    TF_SessionRun(session, NULL, feeds, in_tensors, 1, fetches, oo_tensor, 4, NULL, 0, NULL, status);
    ret = TF_GetCode(status) != TF_OK;
    DEBUG("TF_SessionRun status %c= TF_OK (%s), tm = %ld\n", ret ? '!' : '=', TF_Message(status), time(NULL));
    boxes_data = TF_TensorData(oo_tensor[0]), scores_data = TF_TensorData(oo_tensor[1]), classes_data = TF_TensorData(oo_tensor[2]), count_data = TF_TensorData(oo_tensor[3]);
    count = (int) *(float *)count_data;
    DEBUG("oo_tensor.data (%p, %p, %p, %p), num_detections = %d\n", boxes_data, scores_data, classes_data, count_data, count);

    for(int i = 0; i < 4; i++) {
      TF_Tensor* t = oo_tensor[i];
      int nbytes = TF_TensorByteSize(t);
      switch(TF_NumDims(oo_tensor[i])) {
      case 1:
      DEBUG("oo_tensor[%d].name = %s (%d)", i, tensor_name[i], TF_Dim(t, 0));
      break;
      case 2:
      DEBUG("oo_tensor[%d].name = %s (%d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1));
      break;
      case 3:
      DEBUG("oo_tensor[%d].name = %s (%d, %d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1), TF_Dim(t, 2));
      break;
      case 4:
      DEBUG("oo_tensor[%d].name = %s (%d, %d, %d, %d)", i, tensor_name[i], TF_Dim(t, 0), TF_Dim(t, 1), TF_Dim(t, 2), TF_Dim(t, 3));
      break;
      }
      DEBUG(" type = %x, bytesize = %d\n", TF_TensorType(t), nbytes);
    }

    for(int i = 0; i < count; i++, scores_data++, classes_data++) {
      float scores;
      float x0, y0, x1, y1;
      int classes;
      scores = *(float *) scores_data;
      classes = (int) *(float *) classes_data;
      x0 = *(float *) boxes_data, y0 = *(float *) (boxes_data + 1), x1 = *(float *) (boxes_data + 2), y1 = *(float *) (boxes_data + 3);
      x0 *= width, x1 *= width;
      y0 *= height, y1 *= height;

      if(scores > SCORES_PASS) {
        DEBUG("oo_tensor[%d], scores = %.2f, classes %d, (%.2f, %.2f, %.2f, %.2f)\n", i, scores, classes, x0, y0, x1, y1);
      }
      boxes_data += 4;
    }

    TF_DeleteTensor(oo_tensor[0]);
    TF_DeleteTensor(oo_tensor[1]);
    TF_DeleteTensor(oo_tensor[2]);
    TF_DeleteTensor(oo_tensor[3]);
  }

  gettimeofday(&tm, NULL);
  etm = tm.tv_sec * 1000LL + tm.tv_usec/1000;
  DEBUG("do_image_detection %s time = %lld msec\n\n", jpegfn, etm-stm);

  TF_DeleteStatus(status);
}

void dump_all_operation(TF_Graph* graph) {
  TF_Operation* oper = NULL;
  size_t pos = 0;

  while ((oper = TF_GraphNextOperation(graph, &pos)) != NULL) {
    DEBUG("OP.%d: op.name = %s, op.type = %s, op.device = %s, op.output.num = %d\n", pos, TF_OperationName(oper), TF_OperationOpType(oper), TF_OperationDevice(oper), TF_OperationNumOutputs(oper));
  }
}

int usbcamera_open(const char* usbcam) {
  int fd = 0;
  struct stat st;

  if(stat(usbcam, &st) == -1) {
    DEBUG(" USB CAMERA %s isn't available\n", usbcam);
  }
  if (!S_ISCHR(st.st_mode)) {
    DEBUG(" %s isn't USB CAMERA\n", usbcam);
  }

  fd = v4l2_open(usbcam, O_RDWR /* required */ | O_NONBLOCK, 0);
  DEBUG("*usbcamera_open.v4l2_open fd = %d\n", fd);

  return fd;
}

typedef struct camera_param {
  int cid;
  int val;
  const char* name;
} camera_param_t;

static camera_param_t usb_camera_param[] = {
  //{V4L2_CID_BRIGHTNESS, -17, "V4L2_CID_BRIGHTNESS"},
  {V4L2_CID_BRIGHTNESS, 17, "V4L2_CID_BRIGHTNESS"},
  {V4L2_CID_FOCUS_AUTO, 1, "V4L2_CID_FOCUS_AUTO"},
  {V4L2_CID_FOCUS_ABSOLUTE, 95, "V4L2_CID_FOCUS_ABSOLUTE"},
  {V4L2_CID_EXPOSURE_AUTO, 1, "V4L2_CID_EXPOSURE_AUTO"},
  {V4L2_CID_EXPOSURE_ABSOLUTE, 118, "V4L2_CID_EXPOSURE_ABSOLUTE"},
  {0, 0, NULL}
};

static void usbcamera_start(gotensor_t* go) {
  unsigned int i;
  enum v4l2_buf_type type;
  struct v4l2_control ctrl;
  int ret;
  
  for(i = 0; usb_camera_param[i].cid != 0; i++) {
    ctrl.id = usb_camera_param[i].cid;
    ctrl.value = usb_camera_param[i].val;
    ret = xioctl(go->vfd, VIDIOC_S_CTRL, &ctrl);
    DEBUG("VIDIOC_S_CTRL.%s ret = %d\n", usb_camera_param[i].name, ret);
  }

  for (i = 0; i < go->n_buffers; ++i) {
  	struct v4l2_buffer buf = {0};
  
  	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  	buf.memory = V4L2_MEMORY_MMAP;
  	buf.index = i;
  
  	if (-1 == xioctl(go->vfd, VIDIOC_QBUF, &buf)) {
      DEBUG("usbcamera_start.xioctl VIDIOC_QBUF failed for index(%d)\n", i);
    }
  }
  
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == xioctl(go->vfd, VIDIOC_STREAMON, &type)) {
    DEBUG("xioctl VIDIOC_STREAMON failed\n");
  }
  DEBUG("xioctl VIDIOC_STREAMON n_buffers = %d\n", go->n_buffers);
}

static void usbcamera_stop(int vfd) {
  enum v4l2_buf_type type;

  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (-1 == xioctl(vfd, VIDIOC_STREAMOFF, &type)) {
    DEBUG("xioctl VIDIOC_STREAMOFF failed\n");
  }
  DEBUG("xioctl VIDIOC_STREAMOFF\n");
}

static int v4l2_mmap_init(gotensor_t* go) {
  struct v4l2_requestbuffers req = {0};
  int idx = 0, ret = 0;
  
  req.count = 2;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  
  if (-1 == xioctl(go->vfd, VIDIOC_REQBUFS, &req)) {
  	if (EINVAL == errno) {
      fprintf(stderr, "%s does not support memory mapping\n", go->camera);
  	} else {
      DEBUG("v4l2_mmap_init.xioctl VIDIOC_REQBUFS error = %s\n", strerror(errno));
  	}
    ret = -1;
  }

  DEBUG("VIDIOC_REQBUFS req.count = %d\n", req.count);
  if (req.count < 2) {
  	fprintf(stderr, "Insufficient buffer memory on %s\n", go->camera);
    ret = -1;
  }
  
  go->buffers = calloc(req.count, sizeof(struct yuv_buffer));
  DEBUG("v4l2_mmap_init buffers = %p\n", go->buffers);
  
  for (idx = 0; idx < req.count; ++idx) {
  	struct v4l2_buffer buf = {0};
  
  	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  	buf.memory = V4L2_MEMORY_MMAP;
  	buf.index = idx;
  
  	if (-1 == xioctl(go->vfd, VIDIOC_QUERYBUF, &buf)) {
      DEBUG("v4l2_mmap_init.xioctl VIDIOC_QUERYBUF failed\n");
    }
  
  	go->buffers[idx].length = buf.length;
  	go->buffers[idx].start = v4l2_mmap(NULL /* start anywhere */, buf.length, PROT_READ | PROT_WRITE /* required */, MAP_SHARED /* recommended */, go->vfd, buf.m.offset);
  	if (MAP_FAILED == go->buffers[idx].start) {
      DEBUG("v4l2_mmap_init.v4l2_mmap failed for buffers[%d], buf.length = %d\n", idx, buf.length);
    }
  }
  go->yuvimg = calloc(go->buffers[0].length, sizeof(unsigned char));
  //if(cudaSuccess != cudaMalloc(&go->yuvimg_gpu, sizeof(unsigned char) * (go->buffers[0].length))) {
  //  DEBUG("cudaMalloc yuvimg failed\n");
  //  go->yuvimg = calloc(go->buffers[0].length, sizeof(unsigned char));
  //  go->yuvimg_gpu = NULL;
  //}
  go->n_buffers = req.count;

  return ret;
}

static int usbcamera_setup(gotensor_t* go) {
  struct v4l2_capability cap;
  struct v4l2_cropcap cropcap = {0};
  struct v4l2_crop crop;
  struct v4l2_format fmt = {0};
  struct v4l2_streamparm frameint = {0};
  unsigned int min;
  int ret = 0;
  
  if (-1 == xioctl(go->vfd, VIDIOC_QUERYCAP, &cap)) {
  	if (EINVAL == errno) {
      fprintf(stderr, "%s is no V4L2 device\n", go->camera);
  	}
    ret = -1;
  }
  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    fprintf(stderr, "%s is no video capture device\n", go->camera);
  }
  if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
    fprintf(stderr, "%s does not support streaming i/o\n", go->camera);
  }

  /* Select video input, video standard and tune here. */
  cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (0 == xioctl(go->vfd, VIDIOC_CROPCAP, &cropcap)) {
  	crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  	crop.c = cropcap.defrect; /* reset to default */
  
  	if (-1 == xioctl(go->vfd, VIDIOC_S_CROP, &crop)) {
      //DEBUG("xioctl VIDIOC_S_CROP errno = %s\n", strerror(errno));
  	}
  } else {
  	/* Errors ignored. */
  }
  
  //v4l2_format
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = 1280;
  fmt.fmt.pix.height = 720;
  fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
  //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  
  if (-1 == xioctl(go->vfd, VIDIOC_S_FMT, &fmt)) {
    fprintf(stderr, "xioctl VIDIOC_S_FMT failed\n");
  }
  
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUV420) {
  //if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
  	fprintf(stderr,"camera didn't accept YUV420 format.\n");
  }
  DEBUG("USB Camera accept format V4L2_PIX_FMT_YUV420, width = %d, height = %d \n", fmt.fmt.pix.width, fmt.fmt.pix.height);
  
  go->img_width = fmt.fmt.pix.width;
  go->img_height = fmt.fmt.pix.height;
  
  /* Attempt to set the frame interval. */
  frameint.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  frameint.parm.capture.timeperframe.numerator = 1;
  frameint.parm.capture.timeperframe.denominator = UC_FPS;
  if (-1 == xioctl(go->vfd, VIDIOC_S_PARM, &frameint))
    fprintf(stderr,"Unable to set frame interval.\n");
  
  /* Buggy driver paranoia. */
  min = fmt.fmt.pix.width * 2;
  if (fmt.fmt.pix.bytesperline < min)
  	fmt.fmt.pix.bytesperline = min;
  min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
  if (fmt.fmt.pix.sizeimage < min)
  	fmt.fmt.pix.sizeimage = min;
  

  ret = v4l2_mmap_init(go);
  return ret;
}

void usbcamera_close(gotensor_t* go) {
  for (int i = 0; i < go->n_buffers; ++i) {
    if (-1 == v4l2_munmap(go->buffers[i].start, go->buffers[i].length)) {
      DEBUG("usbcamera_close.v4l2_munmap failed fo buf[%d]\n", i);
    }
  }
  if(go->yuvimg != NULL) {
    free(go->yuvimg);
  }
  if(go->yuvimg_gpu != NULL) {
    cudaFree(go->yuvimg_gpu);
  }

  free(go->buffers);
  v4l2_close(go->vfd);
}

#if 0
static void imageProcess(void* p, int width, int height, struct timeval timestamp) {
  int nDstStep = width * 3;
  int nSrcStep = width * 2;
  int nbytes = nDstStep * height * sizeof(unsigned char);
  unsigned char* dst = malloc(nbytes);
  NppiSize roi = {0, 0};
  const Npp8u* pSrc[3] = {NULL, NULL, NULL};
  int rSrcStep[3] = {0, 0, 0};
  NppStatus npps = 0;
  int ystride = width * height;
  int uvstride = (width * height)/4;
  unsigned char *base_py = p;
  unsigned char *base_pu = p + ystride;
  unsigned char *base_pv = p + ystride + uvstride;
   
  pSrc[0] = base_py, pSrc[1] = base_pu, pSrc[2] = base_pv;
  //pSrc[0] = base_pv, pSrc[1] = base_pu, pSrc[2] = base_py;
  //rSrcStep[0] = height * width, rSrcStep[1] = (height*width)+(height*width)/4, rSrcStep[2] = 0;
  rSrcStep[0] = width, rSrcStep[1] = width/2, rSrcStep[2] = width/2;
  //rSrcStep[2] = width, rSrcStep[1] = width/2, rSrcStep[0] = width/2;

  //printf("NPP_SIZE_ERROR = %x, NPP_STEP_ERROR = %x\n", NPP_SIZE_ERROR, NPP_STEP_ERROR);
  if(dst != NULL) {
    char bmpfn[256] = {0};

    snprintf(bmpfn, 256, "/tmp/%ld.bmp", time(NULL));
    if(access(bmpfn, F_OK) != 0) {
      yuv420_rgb24_sseu(width, height, base_py, base_pu, base_pv, width, width/2, dst, nDstStep, YCBCR_JPEG);
    } else {
      DEBUG("%s is exist, ignore this image frame\n", bmpfn);
    }
    //roi.width = width/2, roi.height = height/2;
    //roi.width = 0, roi.height = 0;
    //npps = nppiYUVToRGB_8u_C3R(p, nSrcStep, dst, nDstStep, roi);
    //npps = nppiYUV422ToRGB_8u_C2C3R(p, nSrcStep, dst, nDstStep, roi);
	//npps = nppiYUV420ToRGB_8u_P3C3R(pSrc, rSrcStep, dst, nDstStep, roi);
    //nppiYUV420ToBGR_8u_P3C3R(pSrc, rSrcStep, dst, nDstStep, roi);
    //printf("save_bitmap to /tmp/001.bmp, width = %d height = %d, nbytes = %d, npps = %x\n", width, height, nbytes, npps);
    free(dst);
  }
}

static void _uc_signal_cb (evutil_socket_t fd, short event, void *arg) {
  gotensor_t* go = (gotensor_t *)arg;

  DEBUG("got signal %x event, fd = %d\n", event, fd);
  event_base_loopbreak(go->base);
}

static void _uc_read_cb(evutil_socket_t fd, short event, void *arg) {
  struct v4l2_buffer buf = {0};
  gotensor_t* go = (gotensor_t *)arg;

  DEBUG(" _uc_read_cb event = %x, fd = %d\n", event, fd);
  if(event & EV_READ) {
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
      DEBUG("_uc_read_cb *xioctl VIDIOC_DQBUF error = %s errno = %x, EIO = %x, EAGAIN = %x\n", strerror(errno), errno, EIO, EAGAIN);
      if(errno == EAGAIN) {
        return;
      }
    }
    
    if(buf.index < go->n_buffers) {
      imageProcess(go->buffers[buf.index].start, go->image_width, go->image_height, buf.timestamp);
    }
    DEBUG("VIDIOC_DQBUF buf.index = %d, n_buffers = %d\n", buf.index, go->n_buffers);
    
    if (-1 == xioctl(fd, VIDIOC_QBUF, &buf)) {
    	DEBUG("_uc_read_cb **xioctl VIDIOC_QBUF error = %s", strerror(errno));
    }
  }
}

static void tfoutput_cairo_new(gotensor_t* go) {
  int nstride = cairo_format_stride_for_width (CAIRO_FORMAT_RGB24, go->img_width);

  go->surface = cairo_image_surface_create_for_data(go->rgbbuf, CAIRO_FORMAT_RGB24, go->img_width, go->img_height, nstride);
}
#endif

static unsigned char* do_yuvimg_detection (gotensor_t* go, int width, int height, unsigned long* jpegSize) {
  int nstride = width * 3;
  int nbytes = nstride * height * sizeof(unsigned char);

  int ystride = width * height;
  int uvstride = (width * height)/4;
  unsigned char *base_py = go->yuvimg;
  unsigned char *base_pu = go->yuvimg + ystride;
  unsigned char *base_pv = go->yuvimg + ystride + uvstride;
  unsigned char *jpegBuf = NULL;
  //unsigned char *dst = malloc(nbytes); 
  void* dst = NULL;
  if(cudaSuccess != cudaHostAlloc(&dst, nbytes, cudaHostAllocMapped)) {
    DEBUG("cudaHostAlloc for rgbimg failed \n");
  }

  struct timeval tm;
  long long stm = 0, etm = 0;

  gettimeofday(&tm, NULL);
  stm = tm.tv_sec * 1000LL + tm.tv_usec/1000;

  if(dst != NULL) {
    unsigned char* rgbbuf_to = go->rgbbuf;
    unsigned char* rgbbuf_from = dst;

    pthread_mutex_lock(&go->mutex);
    yuv420_rgb24_std(width, height, base_py, base_pu, base_pv, width, width/2, dst, nstride, YCBCR_JPEG);
    pthread_mutex_unlock(&go->mutex);
    for(int i = 0; i < height; i++) {
      for(int j = 0; j < width; j++, rgbbuf_from += 3) {
        *rgbbuf_to++ = *(rgbbuf_from+2);
        *rgbbuf_to++ = *(rgbbuf_from+1);
        *rgbbuf_to++ = *(rgbbuf_from+0);
        *rgbbuf_to++ = 0;
      }
    }

    //save RGB buffer to bmp file
    //tjSaveImage("/tmp/001.bmp", dst, width, nstride, height, TJPF_RGB, 0);
    {
      int ret = 0;
      tjhandle tjInstance = tjInitCompress();

      ret = tjCompress2(tjInstance, dst, width, nstride, height, TJPF_RGB, &jpegBuf, jpegSize, TJSAMP_420, 90, 0);
      if(ret < 0) {
        DEBUG("Compress RGB Image Buffer to JPEG Buffer failed\n");
      }
      tjDestroy(tjInstance);
    }
    do_rgbimg_detection(go, dst, nbytes, width, height);
  }

  gettimeofday(&tm, NULL);
  etm = tm.tv_sec * 1000LL + tm.tv_usec/1000;
  DEBUG("do_yuvimg_detection time = %lld msec\n\n", etm-stm);
  return jpegBuf;
}

#if 0
static unsigned char* do_yuvimg_detection_gpu (gotensor_t* go, int width, int height, unsigned long* jpegSize) {
  int nstride = width * 3;
  int nbytes = nstride * height * sizeof(unsigned char);

  int ystride = width * height;
  int uvstride = (width * height)/4;
  unsigned char *base_py = go->yuvimg_gpu;
  unsigned char *base_pu = go->yuvimg_gpu + ystride;
  unsigned char *base_pv = go->yuvimg_gpu + ystride + uvstride;
  unsigned char *jpegBuf = NULL;
   
  void* dst = NULL;
  if(cudaSuccess != cudaMalloc(&dst, nbytes)) {
    DEBUG("cudaMalloc for rgbimg GPU buffer failed\n");
  }

  struct timeval tm;
  long long stm = 0, etm = 0;

  gettimeofday(&tm, NULL);
  stm = tm.tv_sec * 1000LL + tm.tv_usec/1000;

  if(dst != NULL) {
  int nDstStep = width * 3;
  int nSrcStep = width * 2;
  int nbytes = nDstStep * height * sizeof(unsigned char);
  unsigned char* dst = malloc(nbytes);
  NppiSize roi = {0, 0};
  const Npp8u* pSrc[3] = {NULL, NULL, NULL};
  int rSrcStep[3] = {0, 0, 0};
  NppStatus npps = 0;
  int ystride = width * height;
  int uvstride = (width * height)/4;
  unsigned char *base_py = p;
  unsigned char *base_pu = p + ystride;
  unsigned char *base_pv = p + ystride + uvstride;
   
  pSrc[0] = base_py, pSrc[1] = base_pu, pSrc[2] = base_pv;
  //pSrc[0] = base_pv, pSrc[1] = base_pu, pSrc[2] = base_py;
  //rSrcStep[0] = height * width, rSrcStep[1] = (height*width)+(height*width)/4, rSrcStep[2] = 0;
  rSrcStep[0] = width, rSrcStep[1] = width/2, rSrcStep[2] = width/2;
  //rSrcStep[2] = width, rSrcStep[1] = width/2, rSrcStep[0] = width/2;

    roi.width = width, roi.height = height;
    pthread_mutex_lock(&go->mutex);
    npps = nppiYUV420ToRGB_8u_P3C3R(pSrc, rSrcStep, dst, nDstStep, roi);
    pthread_mutex_unlock(&go->mutex);
  }
  do_rgbimg_detection(go, dst, nbytes, width, height);

  gettimeofday(&tm, NULL);
  etm = tm.tv_sec * 1000LL + tm.tv_usec/1000;
  DEBUG("do_yuvimg_detection time = %lld msec\n\n", etm-stm);
  return jpegBuf;
}
#endif

static int gotensor_get_detection (void *cls, struct MHD_Connection *connection, const char *url, const char *method, const char *version, const char *upload_data, size_t *upload_data_size, void **ptr) {
  static int aptr;
  gotensor_t* go = (gotensor_t *)cls;
  struct MHD_Response *response;
  int ret;
  (void)url;               /* Unused. Silent compiler warning. */
  (void)version;           /* Unused. Silent compiler warning. */
  (void)upload_data;       /* Unused. Silent compiler warning. */
  (void)upload_data_size;  /* Unused. Silent compiler warning. */
  char myfn[1024] = {0};
  char* cttype = "image/jpeg";
  int fetch_image = 1;

  if (0 != strcmp (method, "GET"))
    return MHD_NO;              /* unexpected method */

  if (&aptr != *ptr) {
      /* do never respond on first call */
      *ptr = &aptr;
      return MHD_YES;
  }

  *ptr = NULL;                  /* reset when done */
  DEBUG("gotensor_get_detection url = %s, method = %s, version = %s\n", url, method, version);

  if(strcmp(url, "/get/tensorflow/camera0") == 0) {
    unsigned char md5sum[MD5_DIGEST_LENGTH] = {0};
    char md5string[33] = {0};
    char sqlcmd[1024] = {0};
    char* errmsg = NULL;
    unsigned char* jpegBuf = NULL;
    unsigned long  jpegSize;
    cairo_status_t e;
    FILE* fp = NULL;
    char* ojptr = go->output_json;
    g_GET_reqcnt ++;

    fetch_image = 0;
    image_detection_setup(go);
    jpegBuf = do_yuvimg_detection(go, go->img_width, go->img_height, &jpegSize);
    MD5(go->rgbbuf, go->rgbsize, md5sum);
    snprintf(md5string, 33, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x", md5sum[0], md5sum[1], md5sum[2], md5sum[3], md5sum[4], md5sum[5], md5sum[6], md5sum[7], md5sum[8], md5sum[9], md5sum[10], md5sum[11],md5sum[12], md5sum[13], md5sum[14], md5sum[15]);
    //strcat(go->output_json, md5string, );
    while(*ojptr++ != '\0') {
      if(*ojptr == '}' && *(ojptr -1) == ']') {
        *ojptr++ = ',';
        sprintf(ojptr, "\"md5\":\"%s\"}", md5string);
        break;
      }
    }
    //DEBUG("output_json = %s\n", go->output_json);
    response = MHD_create_response_from_buffer (strlen (go->output_json),
                            (void *) go->output_json,
                            MHD_RESPMEM_PERSISTENT);
    MHD_add_response_header(response, MHD_HTTP_HEADER_CONTENT_TYPE, "application/json");
    ret = MHD_queue_response (connection, MHD_HTTP_OK, response);
    MHD_destroy_response (response);

    if(jpegBuf != NULL) {
      snprintf(myfn, 1024, "/tmp/usbcamera/%s_o.jpg", md5string);
      fp = fopen(myfn, "wb");
      if(fp != NULL) {
        fwrite(jpegBuf, jpegSize, 1, fp);
        fclose(fp);
        DEBUG("Write USB Camera image to %s\n", myfn);
        //if(go->db != NULL) {
        //  snprintf(sqlcmd, 1024, "INSERT INTO output VALUES(%ld, \"%s\", \"%s\");", time(NULL), md5string, go->output_csv);
        //  sqlite3_exec(go->db, sqlcmd, NULL, NULL, &errmsg);
        //}
      }
      free(jpegBuf);
    }

#if 1
    myfn[48] = 't';
    e = cairo_image_surface_write_to_jpeg(go->surface, myfn, 90);
    if(CAIRO_STATUS_SUCCESS != e) {
      DEBUG("cairo_image_surface_write_to_jpeg failed, e = %x\n", e);
    }
    DEBUG("Write tensorflow output to %s\n", myfn);
#endif
  } else if(strncmp(url, "/get/image/0/", 13) == 0) {
    snprintf(myfn, 1024, "/tmp/usbcamera/%s_o.jpg", url+13);
  } else if(strncmp(url, "/get/image/1/", 13) == 0) {
    snprintf(myfn, 1024, "/tmp/usbcamera/%s_t.jpg", url+13);
  } else if(strncmp(url, "/favicon.ico", 12) == 0) {
    snprintf(myfn, 1024, "favicon.png");
    cttype = "image/png";
  } else if(strncmp(url, "/tfoutput.jpg", 13) == 0) {
    snprintf(myfn, 1024, "tfoutput.jpg");
  } else if(strncmp(url, "/get/status", 11) == 0) {
    char statbuf[2048] = {0};
    if(go->camera == USB_CAMERA_NULL) {
      snprintf(statbuf, 2048, "{\"status\":200,\"Camera\":0,\"PB\":\"%s\"}", go->pbfn);
    } else {
      snprintf(statbuf, 2048, "{\"status\":200,\"Camera\":1,\"PB\":\"%s\"}", go->pbfn);
    }
    response = MHD_create_response_from_buffer (strlen (statbuf),
                            (void *) statbuf,
                            MHD_RESPMEM_PERSISTENT);
    MHD_add_response_header(response, MHD_HTTP_HEADER_CONTENT_TYPE, "application/json");
    MHD_queue_response (connection, MHD_HTTP_OK, response);
    MHD_destroy_response (response);
  } else {
    snprintf(myfn, 1024, "index.html");
    cttype = "text/html";
  }

  if(fetch_image == 1) {
    FILE* fp = fopen(myfn, "rb");
    if(fp != NULL) {
      fseek(fp, 0, SEEK_END);
      int nsize = ftell(fp);
      DEBUG("try to fetch %s, fp = %p, nsize = %d \n", myfn, fp, nsize);
      fseek(fp, 0, SEEK_SET);
      response = MHD_create_response_from_fd(nsize, fileno(fp));
      MHD_add_response_header(response, MHD_HTTP_HEADER_CONTENT_TYPE, cttype);
      ret = MHD_queue_response (connection, MHD_HTTP_OK, response);
      MHD_destroy_response (response);
    }
    DEBUG("fetch_image %s DONE, fp = %p, g_GET_reqcnt = %d \n", myfn, fp, g_GET_reqcnt);
  }

  return ret;
}

static int gotensor_new(gotensor_t* go, TF_Graph* graph) {
  int fd = -1;

  go->httpd = MHD_start_daemon(MHD_USE_AUTO | MHD_USE_INTERNAL_POLLING_THREAD | MHD_USE_ERROR_LOG,
              8080, NULL, NULL, &gotensor_get_detection, go,
              MHD_OPTION_CONNECTION_TIMEOUT, (unsigned int) 10,
              MHD_OPTION_END);

  if(access("/tmp/usbcamera", F_OK) != 0) {
    mkdir("/tmp/usbcamera", 0777);
  }

  if(access("/tmp/yuvimg", F_OK) != 0) {
    mkdir("/tmp/yuvimg", 0777);
  }

  go->tfStatus = TF_NewStatus();
  go->tfHandle = NULL;
  pthread_mutex_init(&go->mutex, NULL);
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, opts, go->tfStatus);
  TF_DeleteSessionOptions(opts);

  //go->base = event_base_new();
  go->tfGraph = graph;
  go->tfSession = session;
  if(sqlite3_open(TF_OUTPUT_DB, &go->db) != 0) {
    DEBUG("open %s failed\n", TF_OUTPUT_DB);
  } else {
    char* errmsg = NULL;
    int ret = 0;
    ret = sqlite3_exec(go->db, "CREATE TABLE IF NOT EXISTS output (tm INTEGER, key CHAR(32), json TEXT);", NULL, NULL, &errmsg);
    DEBUG("ret = %d, errmsg = %s\n", ret, errmsg);
  }

  go->inop = TF_GraphOperationByName(graph, "image_tensor");
  go->oo_scores = TF_GraphOperationByName(graph, "detection_scores");
  go->oo_boxes = TF_GraphOperationByName(graph, "detection_boxes");
  go->oo_classes = TF_GraphOperationByName(graph, "detection_classes");
  go->oo_count = TF_GraphOperationByName(graph, "num_detections");

  if(getenv("USBCAMERA")) {
    fd = usbcamera_open(getenv("USBCAMERA"));
    go->camera = getenv("USBCAMERA");
  }
  if(fd < 0) {
    fd = usbcamera_open(USB_CAMERA_DEV0);
    if(fd < 0) {
      fd = usbcamera_open(USB_CAMERA_DEV1);
      if(fd > 0)
        go->camera = (char *)USB_CAMERA_DEV1;
    } else {
      go->camera = (char* )USB_CAMERA_DEV0;
    }
  }
  if(fd > 0) {
    go->vfd = fd;
    if(usbcamera_setup(go) == 0) {
      int nstride = cairo_format_stride_for_width (CAIRO_FORMAT_RGB24, go->img_width);
      //go->ucev = event_new(go->base, fd, EV_READ|EV_PERSIST, _uc_read_cb, go);
      //event_add(go->ucev, NULL);

      //go->int_sigev = evsignal_new(go->base, SIGINT, _uc_signal_cb, go);
      //event_add(go->int_sigev, NULL);

      go->rgbsize = nstride * go->img_height;
      //go->rgbbuf = malloc(go->rgbsize);
      if(cudaSuccess != cudaHostAlloc(&go->rgbbuf, go->rgbsize, cudaHostAllocMapped)) {
        DEBUG("cudaHostAlloc rgbbuf failed\n");
      }
      go->surface = cairo_image_surface_create_for_data(go->rgbbuf, CAIRO_FORMAT_RGB24, go->img_width, go->img_height, nstride);
      if(go->autolabel_imgdir == NULL) {
        usbcamera_start(go);
      }
    }
  }

  return fd;
}

static void gotensor_exec(gotensor_t* go) {
  //event_base_dispatch(go->base);
  for (;;) {
  	fd_set fds;
  	struct timeval tv;
  	int r;
  
  	FD_ZERO(&fds);
  	FD_SET(go->vfd, &fds);
  
  	/* Timeout. */
  	tv.tv_sec = 1;
  	tv.tv_usec = 0;
  
  	r = select(go->vfd + 1, &fds, NULL, NULL, &tv);

    //DEBUG("gotensor_exec.select.r = %d, fd = %d, errno = %x, EINTR = %x\n", r, go->vfd, errno, EINTR);
  	if (EINTR == errno) {
      usleep(200);
      continue;
    }

    {
	  struct v4l2_buffer buf = {0};

      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;

      if (-1 == xioctl(go->vfd, VIDIOC_DQBUF, &buf)) {
        //DEBUG("_uc_read_cb *xioctl VIDIOC_DQBUF error = %s errno = %x, EIO = %x, EAGAIN = %x\n", strerror(errno), errno, EIO, EAGAIN);
        if(errno == EAGAIN) {
          return;
        }
      }
      
      if(buf.index < go->n_buffers) {
        //DEBUG("buf.index = %d, go->n_buffers = %d, buf.length = %d\n", buf.index, go->n_buffers, go->buffers[buf.index].length);
        //imageProcess(go->buffers[buf.index].start, go->image_width, go->image_height, buf.timestamp);
        pthread_mutex_lock(&go->mutex);
        if(go->yuvimg_gpu) {
          if(cudaSuccess != cudaMemcpy(go->yuvimg_gpu, go->buffers[buf.index].start, go->buffers[buf.index].length, cudaMemcpyHostToDevice)) {
            DEBUG("cudaMemcpy webcamera buffer to yuvimg_gpu failed\n");
            go->yuvimg_gpu = NULL;
          }
        } else if(go->yuvimg) {
          int buflen = go->buffers[buf.index].length;
          memcpy(go->yuvimg, go->buffers[buf.index].start, buflen);
          if(g_GET_reqcnt > 0) {
            //FILE* yuvfp = fopen(yuvfn, "wb");
            //fwrite(go->yuvimg, sizeof(unsigned char), buflen, yuvfp);
            //fclose(yuvfp);
            char myfn[1024] = {0};
            snprintf(myfn, 1024, "/tmp/yuvimg/%ld.jpg", time(NULL));
            if(access(myfn, F_OK) != 0) {
              int width = 1280, height = 720;
              int nstride = width * 3;
              int nbytes = nstride * height * sizeof(unsigned char);

              int ystride = width * height;
              int uvstride = (width * height)/4;
              unsigned char *base_py = go->yuvimg;
              unsigned char *base_pu = go->yuvimg + ystride;
              unsigned char *base_pv = go->yuvimg + ystride + uvstride;
              unsigned char *jpegBuf = NULL;
              unsigned char *dst = malloc(nbytes); 

              if(dst != NULL) {
                yuv420_rgb24_std(width, height, base_py, base_pu, base_pv, width, width/2, dst, nstride, YCBCR_JPEG);

                {
                  int ret = 0;
                  unsigned long jpegSize = 0;
                  unsigned char* jpegBuf = NULL;

                  tjhandle tjInstance = tjInitCompress();

                  ret = tjCompress2(tjInstance, dst, width, nstride, height, TJPF_RGB, &jpegBuf, &jpegSize, TJSAMP_420, 90, 0);
                  if(ret < 0) {
                    DEBUG("Compress RGB Image Buffer to JPEG Buffer failed\n");
                  }
                  tjDestroy(tjInstance);
                  if(jpegBuf) {
                    FILE* fp = fopen(myfn, "wb");
                    if(fp != NULL) {
                      fwrite(jpegBuf, jpegSize, 1, fp);
                      fclose(fp);
                      //DEBUG("save yuv buffer to %s\n", myfn);
                    }
                    free(jpegBuf);
                  }
                }
                free(dst);
              } /**if(dst != NULL) */
            }
          } /** if(g_GET_reqcnt > 0) then save yuvimg as jpeg*/
        }
        //DEBUG("cache yuv420 image\n");
        pthread_mutex_unlock(&go->mutex);
      }
      //DEBUG("VIDIOC_DQBUF buf.index = %d, n_buffers = %d\n", buf.index, go->n_buffers);
      
      if (-1 == xioctl(go->vfd, VIDIOC_QBUF, &buf)) {
      	//DEBUG("_uc_read_cb **xioctl VIDIOC_QBUF error = %s", strerror(errno));
      }
    }
  	/* EAGAIN - continue select loop. */
  }
}

static int gotensor_del(gotensor_t* go) {
  //if(go->int_sigev) {
  //  event_free(go->int_sigev);
  //}
  //if(go->ucev) {
  //  event_free(go->ucev);
  //}

  //event_base_free(go->base);

  if(go->surface) {
    cairo_surface_destroy(go->surface);
  }
  if(go->rgbbuf != NULL) {
    //free(go->rgbbuf);
    cudaFreeHost(go->rgbbuf);
  }

  if(go->db != NULL) {
    sqlite3_close(go->db);
  }
  pthread_mutex_destroy(&go->mutex);
  TF_DeleteSession(go->tfSession, go->tfStatus);
  TF_DeleteStatus(go->tfStatus);

  if(go->httpd != NULL) {
    MHD_stop_daemon(go->httpd);
    DEBUG("MHD_stop_daemon DONE\n");
  }
  if(go->vfd > 0) {
    if(go->autolabel_imgdir == NULL) {
      usbcamera_stop(go->vfd);
    } else {
      free(go->autolabel_imgdir);
    }
    usbcamera_close(go);
  }
}

static void generate_annotation(gotensor_t* go, char* imgfn, char* filename, char* folder) {
  xmlDocPtr doc = NULL;
  int nlen = strlen(imgfn);

  if(strncmp(imgfn+nlen-4, ".jpg", 4) == 0) {
    char* xmlfn = strdup(imgfn);

    xmlNodePtr rootnode = NULL, node = NULL;

    doc = xmlNewDoc("1.0");
    rootnode = xmlNewNode(NULL, "annotation");

    xmlDocSetRootElement(doc, rootnode);

    xmlNewChild(rootnode, NULL, "folder", folder);
    xmlNewChild(rootnode, NULL, "filename", filename);
    xmlNewChild(rootnode, NULL, "path", imgfn);

    node = xmlNewChild(rootnode, NULL, "source", NULL);
    xmlNewChild(node, NULL, "database", "Unknown");

    node = xmlNewChild(rootnode, NULL, "size", NULL);
    xmlNewChild(node, NULL, "width", "1280");
    xmlNewChild(node, NULL, "height", "720");
    xmlNewChild(node, NULL, "depth", "3");

    xmlNewChild(rootnode, NULL, "segmented", "0");

    do_jpegimg_detection_ex(go, imgfn, _output_xml_cb, rootnode);

    *(xmlfn + nlen - 1) = 'l';
    *(xmlfn + nlen - 2) = 'm';
    *(xmlfn + nlen - 3) = 'x';
    xmlSaveFormatFileEnc(xmlfn, doc, "UTF-8", 1);
    DEBUG("generate_annotation xmlfn = %s\n", xmlfn);
    xmlFreeDoc(doc);
    free(xmlfn);
  }
}

void auto_label_images(gotensor_t*go){
  FTS* fts = NULL;
  FTSENT *curr = NULL;
  char* ftsargv[] = {NULL, NULL};

  ftsargv[0] = go->autolabel_imgdir;
  fts = fts_open(ftsargv, FTS_NOCHDIR|FTS_PHYSICAL|FTS_XDEV, NULL);
  if(fts != NULL) {
    while(curr = fts_read(fts)) {
      int r1 = -1, r = 0;
      char* folder, *filename, *fnpath;
      struct stat info = {0};
      switch(curr->fts_info) {
      case FTS_NS:
      case FTS_DNR:
      case FTS_ERR:
        //DEBUG("fts_read error: %s,%s, g_printer_daemon_quit = %d\n", curr->fts_accpath, strerror(curr->fts_errno), g_printer_daemon_quit);
        break;
      case FTS_DC:
      case FTS_DOT:
      case FTS_NSOK:
      case FTS_D:
      case FTS_DP:
      case FTS_F:
      case FTS_SL:
      case FTS_SLNONE:
      case FTS_DEFAULT:
        stat(curr->fts_accpath, &info);
        //DEBUG("auto_label_images %s, fts_path = %s, name = %s\n", curr->fts_accpath, curr->fts_path, curr->fts_name);
        if(!S_ISDIR(info.st_mode)) {
          FTSENT* parent = curr->fts_parent;
          if(parent != NULL) {
          DEBUG("try to generate image annotation for %s, directory = %s, %s\n", curr->fts_accpath, parent->fts_name, parent->fts_path);
            generate_annotation(go, curr->fts_path, curr->fts_name, parent->fts_name);
          }
        }
       
        break;
      }
    }
    fts_close(fts);
  }
}

int main (int argc, char* argv[]) {
  int cudaver = 0;
  cuDriverGetVersion(&cudaver);
  DEBUG("tensorflow v%s, CUDA versioncode %d\n", TF_Version(), cudaver);

  const char* pbfn = argv[1];
  gotensor_t go = {0};

  if(argc == 3) {
    if(argv[2] != NULL) {
      int nlen = 0;
      if(access(argv[2], F_OK) == 0) {
        char* ptr = NULL;
        go.autolabel_imgdir = strdup(argv[2]);
        nlen = strlen(go.autolabel_imgdir);
        ptr = go.autolabel_imgdir + nlen -1;
        if(*ptr == '/') {
          *ptr = '\0';
        }
        if(*(ptr - 1) == '/') {
          free(go.autolabel_imgdir);
          DEBUG("autolabel image directory %s is INVALID\n", argv[2]);
          return -2;
        }
      } else {
        DEBUG("autolabel image directory %s is not ACCESSABLE\n", argv[2]);
        return -1;
      }
    }
  }
  go.pbfn = strdup(pbfn);
  go.camera = (char *)USB_CAMERA_NULL;

  if(CUDA_SUCCESS != cuInit(0)) {
    DEBUG("CUDA initialize failed\n");
  }

  TF_Graph *graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
  TF_Buffer* graphdef = load_graph_file_gpu(pbfn);

  //show_tf_devices(graph);
  //dump_all_operation(graph);
  if(graphdef != NULL) {
    TF_GraphImportGraphDef(graph, graphdef, opts, status);
    if(TF_GetCode(status) != TF_OK) {
      DEBUG("import graph %s failed\n", pbfn);
    }
    DEBUG("load graph file %s successful\n", pbfn);

    gotensor_new(&go, graph);

    image_detection_setup(&go);
    //auto_label_images(&go, "/home/tf/mpv-screenshot");
    if(go.autolabel_imgdir != NULL) {
      auto_label_images(&go);
    } else {
      gotensor_exec(&go);
    }

    gotensor_del(&go);
  }

  TF_DeleteStatus(status);
  TF_DeleteBuffer(graphdef);
  TF_DeleteGraph(graph);

  return 0;
}
