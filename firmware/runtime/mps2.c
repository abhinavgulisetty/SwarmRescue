#include <stdint.h>
extern uint32_t _stack_top, _bss_start, _bss_end, _data_start, _data_end, _data_load;
extern int routing_service(void);
#define UART ((volatile uint32_t *)0x40004000u)
int transport_get(void) { while (!(UART[1]&2)) {} return (int)(UART[0]&255); }
void transport_put(uint8_t value) { while (UART[1]&1) {} UART[0]=value; }
void Reset_Handler(void);
__attribute__((section(".vectors"),used))
const uintptr_t vectors[]={(uintptr_t)&_stack_top,(uintptr_t)Reset_Handler};
void Reset_Handler(void) {
    uint32_t *source=&_data_load;
    for (uint32_t *p=&_data_start;p<&_data_end;p++) *p=*source++;
    for (uint32_t *p=&_bss_start;p<&_bss_end;p++) *p=0;
    UART[2]=3;UART[4]=16;
    uint32_t parameters[2]={0x20026,(uint32_t)routing_service()};
    register uint32_t r0 __asm__("r0")=0x20;
    register uint32_t *r1 __asm__("r1")=parameters;
    __asm__ volatile("bkpt 0xab":"+r"(r0):"r"(r1):"memory");
    for (;;) {}
}
