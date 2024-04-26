
#include <assert.h>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>

#include "onnxruntime_c_api.h"

#define BLOCK_LEN (512)
#define STATE_SIZE (128)
#define SAMPLE_RATE (16000)
#define MAX_FRAME (15)

const OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

struct vad_engine{
  float in_audio[BLOCK_LEN]={0};
  int64_t in_sr[1];
  float _h[STATE_SIZE]={0};
  float _c[STATE_SIZE]={0};
  float threshold;
  int max_frame;
  bool triggerd;
  int silence_count;


};
void appendtofile(const char *path, uint8_t *buf, int len){
  static int first=1;
  if (first) {
    first = 0;
    unlink(path);
  }

  int fd = open(path, O_APPEND|O_CREAT|O_RDWR, 0777);
  if(fd<0){return;}
      
  write(fd, buf, len);
  close(fd);
}


int run_inference(OrtSession* session, vad_engine* m_pEngine) {
  
  
  OrtMemoryInfo* memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  const int64_t input_node_dims[2] = {1,BLOCK_LEN}; 
  const int64_t sr_node_dims[1] = {1};
  const int64_t hc_node_dims[3] = {2, 1, 64};

  int is_tensor;
  OrtValue*  ort_input=NULL;
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, m_pEngine->in_audio, BLOCK_LEN*sizeof(float),input_node_dims,
                                                           2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&ort_input));

  assert(ort_input != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(ort_input, &is_tensor));
  assert(is_tensor);
	
	OrtValue*  ort_sr=NULL;
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, m_pEngine->sr,  1*sizeof(int64_t),sr_node_dims,
                                                      1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,&ort_sr));
  assert(ort_sr != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(ort_sr, &is_tensor));
  assert(is_tensor);
  OrtValue*  ort_h=NULL;
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, m_pEngine->_h, STATE_SIZE*sizeof(float), hc_node_dims, 3,
	                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&ort_h));
	assert(ort_h != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(ort_h, &is_tensor));
  assert(is_tensor);
  OrtValue*  ort_c=NULL;
	ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, m_pEngine->_c,STATE_SIZE*sizeof(float), hc_node_dims, 3,
	                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&ort_c));
	assert(ort_c != NULL);
  ORT_ABORT_ON_ERROR(g_ort->IsTensor(ort_c, &is_tensor));
  assert(is_tensor);

	const OrtValue* const ort_inputs[]={ort_input,ort_sr,ort_h,ort_c};

  g_ort->ReleaseMemoryInfo(memory_info);
  const char* input_names[] = {"input", "sr", "h", "c"};
  const char* output_names[] = {"output", "hn", "cn"};
  
  OrtValue*   ort_outputs[]={NULL,NULL,NULL};
	ORT_ABORT_ON_ERROR(g_ort->Run(session,NULL,input_node_names.data(), ort_inputs, 4,output_node_names.data(), 3,ort_outputs));
  int i=0;
  for (i=0;i<3;i++){
    assert(ort_outputs[i] != NULL);
    ORT_ABORT_ON_ERROR(g_ort->IsTensor(ort_outputs[i], &is_tensor));
    assert(is_tensor);
  }

  int ret = 0;
  float *output=NULL;
	ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(ort_outputs[0],(void**)&output));

  float *hn = NULL;
	ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(ort_outputs[1],(void**)&hn));
	memcpy(m_pEngine->_h, hn, STATE_SIZE * sizeof(float));
	float *cn = NULL;
	ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(ort_outputs[2],(void**)&cn));
	memcpy(m_pEngine->_c, cn, STATE_SIZE * sizeof(float));

  if(output[0]>m_pEngine->threshold){
    ret=1;
    m_pEngine->triggerd=true;
    m_pEngine->silence_count=MAX_FRAME;
  }else{
    if(!m_pEngine->triggerd){
      ret=0;
    }else{
      if(output[i]>=(m_pEngine->threshold-0.15)){
        m_pEngine->silence_count=MAX_FRAME;
        ret=1;
      }else{
        m_pEngine->silence_count -=1;
        ret=1;
        if(m_pEngine->silence_count<0){
          ret=0;
          m_pEngine->silence_count=0;
          m_pEngine->triggerd=false;
        }
      }
    }
  }
  for (i=0;i<3;i++){
		g_ort->ReleaseValue(ort_outputs[i]);
				
	}
	g_ort->ReleaseValue(ort_input);
	g_ort->ReleaseValue(ort_sr);
	g_ort->ReleaseValue(ort_h);
	g_ort->ReleaseValue(ort_c);;
  
  return ret;
}


int main(const char* in_file,const char* out_file) {

  vad_engine *m_pEngine;
  m_pEngine=(vad_engine *)malloc(sizeof(vad_engine));

  //set config
  m_pEngine->triggerd=false;
  m_pEngine->max_frame=MAX_FRAME;
  m_pEngine->silence_count=0;
  m_pEngine->threshold=0.5

  m_pEngine->in_sr[0]=SAMPLE_RATE;

  int16_t in_buffer[BLOCK_LEN]={0};
  int16_t out_buffer[BLOCK_LEN]={0};
  

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fprintf("Failed to init ONNX Runtime engine.\n");
    return -1;
  }

  OrtEnv* env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  assert(env != NULL);
  int ret = 0;
  OrtSessionOptions* session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  OrtSession* session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, "silero_vad.onnx", session_options, &session));
  int16_t buffer[BLOCK_LEN]={0};
  int fd=0;
  fd=open(in_file,O_RDONLY);
  if(fd<0){
    fprintf("Failed to open file.\n");
    return -1;

  }
  int fileret;
  int i=0;
  while(1){
    fileret=read(fd,in_buffer,BLOCK_LEN*2);
    if(fileret<BLOCK_LEN*2){
      break;
    }
    for(i=0;i<BLOCK_LEN;i++){
      m_pEngine->in_audio[i]=in_buffer[i]/32767;
    }

    ret = run_inference(session, m_pEngine);
    for(i=0;i<BLOCK_LEN;i++){
      out_buffer[i]=in_buffer[i]*ret;
    }
    appendtofile(out_file,(uint8_t *)out_buffer,BLOCK_LEN*2);

  }
  
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);
  

  free(m_pEngine);

  return ret;
}
