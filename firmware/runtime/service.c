#include <stdint.h>
#include <limits.h>
#include "log_lmdp.h"
#define MAX_N 2048
#define MAX_E 8192
#define MAX_PAYLOAD (20u+10u*MAX_N+8u*MAX_E)
#define MAX_WORK (5*MAX_N+1+MAX_E)
#define REQUEST_MAGIC 0x51524d45u
#define RESPONSE_MAGIC 0x53524d45u
static int32_t rows[MAX_N+1], columns[MAX_E], passive[MAX_E], cost[MAX_N];
static int32_t values_a[MAX_N], workspace[MAX_WORK];
static uint8_t exits[MAX_N], deaths[MAX_N];
extern int transport_get(void);
extern void transport_put(uint8_t value);
static uint32_t input_crc;
static int remaining;
static uint32_t crc_byte(uint32_t crc, uint8_t value) {
    crc ^= value;
    for (int i=0;i<8;i++) crc=(crc>>1)^((crc&1)?0xedb88320u:0u);
    return crc;
}
static int raw_u32(uint32_t *value) {
    *value=0;
    for (int i=0;i<4;i++) {
        int byte=transport_get();if (byte<0) return -1;
        *value |= (uint32_t)(uint8_t)byte << (8*i);
    }
    return 0;
}
static void out_u32(uint32_t value) {
    for (int i=0;i<4;i++) transport_put((uint8_t)(value>>(8*i)));
}
static uint8_t data_byte(void) {
    int value=transport_get();
    if (value<0 || remaining<=0) { remaining=-1;return 0; }
    remaining--;
    input_crc=crc_byte(input_crc,(uint8_t)value);
    return (uint8_t)value;
}
static uint32_t data_u32(void) {
    uint32_t value=0;
    for (int i=0;i<4;i++) value|=(uint32_t)data_byte()<<(8*i);
    return value;
}
static uint32_t output_crc_u32(uint32_t crc,uint32_t value) {
    for (int i=0;i<4;i++) crc=crc_byte(crc,(uint8_t)(value>>(8*i)));
    return crc;
}
static void response(uint32_t sequence,uint32_t status,uint32_t length,uint32_t crc) {
    out_u32(RESPONSE_MAGIC);out_u32(sequence);out_u32(status);out_u32(length);out_u32(crc);
}
int routing_service(void) {
    for (;;) {
        uint32_t magic,command,sequence,length,expected_crc;
        if (raw_u32(&magic)||raw_u32(&command)||raw_u32(&sequence)||raw_u32(&length)||raw_u32(&expected_crc)) return 0;
        if (magic!=REQUEST_MAGIC || length>MAX_PAYLOAD) return 2;
        remaining=(int)length;input_crc=0xffffffffu;
        if (command==0 && length==0 && expected_crc==0) {
#ifdef __arm__
            extern uint32_t _stack_top;
            uint32_t ram=(uint32_t)((uintptr_t)&_stack_top-0x20000000u);
#else
            uint32_t ram=0;
#endif
            uint32_t crc=output_crc_u32(0xffffffffu,1);
            crc=output_crc_u32(crc,MAX_N);crc=output_crc_u32(crc,MAX_E);
            crc=output_crc_u32(crc,ram);
            response(sequence,0,16,crc^0xffffffffu);
            out_u32(1);out_u32(MAX_N);out_u32(MAX_E);out_u32(ram);
            continue;
        }
        if (command==2 && length==0 && expected_crc==0) {
            response(sequence,0,0,0);return 0;
        }
        uint32_t n=0,e=0,limit=0,strategy=0;
        int valid=(command==1 || command==3) && length>=(command==3?16u:12u);
        if (valid) {
            n=data_u32();e=data_u32();limit=data_u32();
            if (command==3) strategy=data_u32();
            valid=n>0 && n<=MAX_N && e<=MAX_E && limit>0 && limit<=10000
                && strategy<=EMBER_LOG_AUTO
                && length==(command==3?20u:16u)+10*n+8*e;
        }
        if (valid) {
            for (uint32_t i=0;i<=n;i++) rows[i]=(int32_t)data_u32();
            for (uint32_t i=0;i<e;i++) columns[i]=(int32_t)data_u32();
            for (uint32_t i=0;i<e;i++) passive[i]=(int32_t)data_u32();
            for (uint32_t i=0;i<n;i++) cost[i]=(int32_t)data_u32();
            for (uint32_t i=0;i<n;i++) exits[i]=data_byte();
            for (uint32_t i=0;i<n;i++) deaths[i]=data_byte();
        }
        while (remaining>0) (void)data_byte();
        if (remaining<0) return 3;
        if ((input_crc^0xffffffffu)!=expected_crc) {
            response(sequence,1,0,0);continue;
        }
        if (!valid) { response(sequence,2,0,0);continue; }
        int32_t *current=values_a;
        ember_log_stats stats;
        if (ember_log_q16_solve((int32_t)n,(int32_t)e,rows,columns,passive,cost,
                exits,deaths,(int32_t)limit,(int32_t)strategy,current,workspace,MAX_WORK,&stats)) {
            response(sequence,3,0,0);continue;
        }
        uint32_t sweeps=stats.sweeps,fixed=stats.fixed_point;
        uint32_t crc=output_crc_u32(0xffffffffu,sweeps);
        crc=output_crc_u32(crc,fixed);
        for (uint32_t i=0;i<n;i++) crc=output_crc_u32(crc,(uint32_t)current[i]);
        for (uint32_t i=0;i<n;i++) crc=output_crc_u32(crc,(uint32_t)ember_log_q16_next((int32_t)i,(int32_t)n,rows,columns,passive,current));
        response(sequence,0,8+8*n,crc^0xffffffffu);
        out_u32(sweeps);out_u32(fixed);
        for (uint32_t i=0;i<n;i++) out_u32((uint32_t)current[i]);
        for (uint32_t i=0;i<n;i++) out_u32((uint32_t)ember_log_q16_next((int32_t)i,(int32_t)n,rows,columns,passive,current));
    }
}
