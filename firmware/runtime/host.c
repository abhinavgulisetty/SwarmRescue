#include <stdint.h>
#include <stdio.h>
extern int routing_service(void);
int transport_get(void) { return getchar(); }
void transport_put(uint8_t value) { putchar(value);fflush(stdout); }
int main(void) { return routing_service(); }
