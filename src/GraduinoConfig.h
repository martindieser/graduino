#pragma once

#include <stdint.h>

#ifndef GRADUINO_CONFIG_H 
#define GRADUINO_CONFIG_H

#if defined(GRADUINO_TINY_MEM)
    using node_id_t = uint8_t;
#else
    using node_id_t = uint16_t;
#endif


#endif