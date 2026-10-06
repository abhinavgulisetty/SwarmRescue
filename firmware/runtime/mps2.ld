ENTRY(Reset_Handler)
MEMORY {
  CODE (rx) : ORIGIN = 0x00000000, LENGTH = 4M
  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 200K
}
SECTIONS {
  .vectors : { KEEP(*(.vectors)) } > CODE
  .text : { *(.text*) *(.rodata*) } > CODE
  .data : {
    _data_start = .; *(.data*) _data_end = .;
  } > RAM AT > CODE
  _data_load = LOADADDR(.data);
  .bss (NOLOAD) : {
    _bss_start = .; *(.bss*) *(COMMON) _bss_end = .;
  } > RAM
  .stack (NOLOAD) : {
    . = ALIGN(8); _stack_bottom = .; . += 16K; _stack_top = .;
  } > RAM
  /DISCARD/ : { *(.comment*) *(.note*) *(.ARM.exidx*) *(.ARM.extab*) }
}
ASSERT(_stack_top <= ORIGIN(RAM) + LENGTH(RAM), "kernel replay exceeds RAM")
