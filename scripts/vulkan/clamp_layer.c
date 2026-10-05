// Vulkan layer that clamps maxMemoryAllocationSize (Maintenance3/Vulkan11) and
// maxBufferSize (Maintenance4/Vulkan13).
//
// Isaac Sim's rtx.scenedb (4.5 and 5.1) computes "buffer size instance limit" as
// maxMemoryAllocationSize / elemSize truncated to 32 bits; a driver that reports UINT64_MAX there
// (595.71, 615.71) overflows it and Kit segfaults at renderer startup.
//
// Env:
//   VKCLAMP_MAX_BUFFER_SIZE    clamp to min(reported, value); default 4292870144,
//                              what driver 580/590 reports for maxMemoryAllocationSize
//   VKCLAMP_FORCE_BUFFER_SIZE  replace the reported value before clamping (repro testing)
//   VKCLAMP_DEBUG              log every property struct queried
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#define EXPORT __attribute__((visibility("default")))
#define MAX_ENTRIES 64

typedef struct {
    void *key;
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkDestroyInstance destroy;
    PFN_vkGetPhysicalDeviceProperties2 props2;
    PFN_vkGetPhysicalDeviceProperties2 props2_khr;
} InstanceEntry;

typedef struct {
    void *key;
    PFN_vkGetDeviceProcAddr gdpa;
    PFN_vkDestroyDevice destroy;
} DeviceEntry;

static InstanceEntry g_instances[MAX_ENTRIES];
static DeviceEntry g_devices[MAX_ENTRIES];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_logged;

static void *dispatch_key(const void *handle) { return *(void *const *)handle; }

static InstanceEntry *find_instance(void *key) {
    for (int i = 0; i < MAX_ENTRIES; ++i)
        if (g_instances[i].key == key) return &g_instances[i];
    return NULL;
}

static DeviceEntry *find_device(void *key) {
    for (int i = 0; i < MAX_ENTRIES; ++i)
        if (g_devices[i].key == key) return &g_devices[i];
    return NULL;
}

static uint64_t env_u64(const char *name, uint64_t fallback, int *present) {
    const char *value = getenv(name);
    if (present) *present = value && *value;
    if (!value || !*value) return fallback;
    return strtoull(value, NULL, 0);
}

static void clamp_chain(VkPhysicalDeviceProperties2 *props) {
    int force = 0;
    uint64_t forced = env_u64("VKCLAMP_FORCE_BUFFER_SIZE", 0, &force);
    uint64_t limit = env_u64("VKCLAMP_MAX_BUFFER_SIZE", 4292870144ull, NULL);
    static int debug_count;
    int debug = getenv("VKCLAMP_DEBUG") && debug_count < 400;
    for (VkBaseOutStructure *s = (VkBaseOutStructure *)props->pNext; s; s = s->pNext) {
        if (debug) { ++debug_count; fprintf(stderr, "[vkclamp] props2 sType=%d\n", (int)s->sType); }
        VkDeviceSize *field = NULL;
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
            field = &((VkPhysicalDeviceMaintenance3Properties *)s)->maxMemoryAllocationSize;
        else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES)
            field = &((VkPhysicalDeviceVulkan11Properties *)s)->maxMemoryAllocationSize;
        else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES)
            field = &((VkPhysicalDeviceMaintenance4Properties *)s)->maxBufferSize;
        else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES)
            field = &((VkPhysicalDeviceVulkan13Properties *)s)->maxBufferSize;
        if (!field) continue;
        VkDeviceSize before = *field;
        VkDeviceSize value = force ? forced : before;
        *field = value > limit ? limit : value;
        if (g_logged < 8) {
            ++g_logged;
            fprintf(stderr, "[vkclamp] %s sType=%d max size %llu -> %llu\n", props->properties.deviceName, (int)s->sType,
                    (unsigned long long)before, (unsigned long long)*field);
        }
    }
}

static VKAPI_ATTR void VKAPI_CALL clamp_GetPhysicalDeviceProperties2(
    VkPhysicalDevice physical_device, VkPhysicalDeviceProperties2 *props) {
    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(dispatch_key(physical_device));
    PFN_vkGetPhysicalDeviceProperties2 next = entry ? (entry->props2 ? entry->props2 : entry->props2_khr) : NULL;
    pthread_mutex_unlock(&g_lock);
    if (!next) return;
    next(physical_device, props);
    clamp_chain(props);
}

static VKAPI_ATTR void VKAPI_CALL clamp_GetPhysicalDeviceProperties2KHR(
    VkPhysicalDevice physical_device, VkPhysicalDeviceProperties2 *props) {
    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(dispatch_key(physical_device));
    PFN_vkGetPhysicalDeviceProperties2 next = entry ? (entry->props2_khr ? entry->props2_khr : entry->props2) : NULL;
    pthread_mutex_unlock(&g_lock);
    if (!next) return;
    next(physical_device, props);
    clamp_chain(props);
}

static VKAPI_ATTR VkResult VKAPI_CALL clamp_CreateInstance(
    const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc, VkInstance *instance) {
    VkLayerInstanceCreateInfo *chain = (VkLayerInstanceCreateInfo *)info->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerInstanceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult result = create(info, alloc, instance);
    if (result != VK_SUCCESS) return result;

    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(NULL);
    if (entry) {
        entry->key = dispatch_key(*instance);
        entry->gipa = gipa;
        entry->destroy = (PFN_vkDestroyInstance)gipa(*instance, "vkDestroyInstance");
        entry->props2 = (PFN_vkGetPhysicalDeviceProperties2)gipa(*instance, "vkGetPhysicalDeviceProperties2");
        entry->props2_khr =
            (PFN_vkGetPhysicalDeviceProperties2)gipa(*instance, "vkGetPhysicalDeviceProperties2KHR");
    }
    pthread_mutex_unlock(&g_lock);
    fprintf(stderr, "[vkclamp] layer active on instance %p\n", (void *)*instance);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL clamp_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(dispatch_key(instance));
    PFN_vkDestroyInstance next = entry ? entry->destroy : NULL;
    if (entry) memset(entry, 0, sizeof(*entry));
    pthread_mutex_unlock(&g_lock);
    if (next) next(instance, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL clamp_CreateDevice(VkPhysicalDevice physical_device,
                                                         const VkDeviceCreateInfo *info,
                                                         const VkAllocationCallbacks *alloc, VkDevice *device) {
    VkLayerDeviceCreateInfo *chain = (VkLayerDeviceCreateInfo *)info->pNext;
    while (chain && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                      chain->function == VK_LAYER_LINK_INFO))
        chain = (VkLayerDeviceCreateInfo *)chain->pNext;
    if (!chain) return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr gipa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    PFN_vkCreateDevice create = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");
    VkResult result = create(physical_device, info, alloc, device);
    if (result != VK_SUCCESS) return result;

    pthread_mutex_lock(&g_lock);
    DeviceEntry *entry = find_device(NULL);
    if (entry) {
        entry->key = dispatch_key(*device);
        entry->gdpa = gdpa;
        entry->destroy = (PFN_vkDestroyDevice)gdpa(*device, "vkDestroyDevice");
    }
    pthread_mutex_unlock(&g_lock);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL clamp_DestroyDevice(VkDevice device, const VkAllocationCallbacks *alloc) {
    pthread_mutex_lock(&g_lock);
    DeviceEntry *entry = find_device(dispatch_key(device));
    PFN_vkDestroyDevice next = entry ? entry->destroy : NULL;
    if (entry) memset(entry, 0, sizeof(*entry));
    pthread_mutex_unlock(&g_lock);
    if (next) next(device, alloc);
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkclamp_GetDeviceProcAddr(VkDevice device, const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkclamp_GetDeviceProcAddr;
    if (!strcmp(name, "vkDestroyDevice")) return (PFN_vkVoidFunction)clamp_DestroyDevice;
    pthread_mutex_lock(&g_lock);
    DeviceEntry *entry = find_device(dispatch_key(device));
    PFN_vkGetDeviceProcAddr next = entry ? entry->gdpa : NULL;
    pthread_mutex_unlock(&g_lock);
    return next ? next(device, name) : NULL;
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkclamp_GetInstanceProcAddr(VkInstance instance, const char *name) {
    if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkclamp_GetInstanceProcAddr;
    if (!strcmp(name, "vkCreateInstance")) return (PFN_vkVoidFunction)clamp_CreateInstance;
    if (!strcmp(name, "vkDestroyInstance")) return (PFN_vkVoidFunction)clamp_DestroyInstance;
    if (!strcmp(name, "vkCreateDevice")) return (PFN_vkVoidFunction)clamp_CreateDevice;
    if (!strcmp(name, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkclamp_GetDeviceProcAddr;
    if (!strcmp(name, "vkDestroyDevice")) return (PFN_vkVoidFunction)clamp_DestroyDevice;
    if (!strcmp(name, "vkGetPhysicalDeviceProperties2"))
        return (PFN_vkVoidFunction)clamp_GetPhysicalDeviceProperties2;
    if (!strcmp(name, "vkGetPhysicalDeviceProperties2KHR"))
        return (PFN_vkVoidFunction)clamp_GetPhysicalDeviceProperties2KHR;
    if (instance == VK_NULL_HANDLE) return NULL;
    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(dispatch_key(instance));
    PFN_vkGetInstanceProcAddr next = entry ? entry->gipa : NULL;
    pthread_mutex_unlock(&g_lock);
    return next ? next(instance, name) : NULL;
}
