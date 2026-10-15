// Vulkan layer that clamps maxMemoryAllocationSize (Maintenance3/Vulkan11) and
// maxBufferSize (Maintenance4/Vulkan13).
//
// Isaac Sim's rtx.scenedb (4.5 and 5.1) computes "buffer size instance limit" as
// maxMemoryAllocationSize / elemSize truncated to 32 bits; a driver that reports UINT64_MAX there
// (595.71, 615.71) overflows it and Kit segfaults at renderer startup.
//
// It also hides the physical devices CUDA cannot see: when CUDA_VISIBLE_DEVICES lists GPU UUIDs (GPU-...), only
// those devices are enumerated, so Kit's renderer and CUDA agree on one card instead of Kit opening every GPU.
//
// Env:
//   CUDA_VISIBLE_DEVICES       UUID entries (full or prefix) limit enumeration; index entries change nothing
//   VKCLAMP_MAX_BUFFER_SIZE    clamp to min(reported, value); default 4292870144,
//                              what driver 580/590 reports for maxMemoryAllocationSize
//   VKCLAMP_FORCE_BUFFER_SIZE  replace the reported value before clamping (repro testing)
//   VKCLAMP_DEBUG              log every property struct queried
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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
    PFN_vkEnumeratePhysicalDevices enumerate;
    PFN_vkEnumeratePhysicalDeviceGroups groups;
    PFN_vkEnumeratePhysicalDeviceGroups groups_khr;
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

// 1 if CUDA_VISIBLE_DEVICES names this device by UUID or has no UUID entries; 0 if it lists UUIDs without it
static int cuda_visible(InstanceEntry *entry, VkPhysicalDevice physical_device) {
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (!cvd || !strstr(cvd, "GPU-")) return 1;
    PFN_vkGetPhysicalDeviceProperties2 props2 = entry->props2 ? entry->props2 : entry->props2_khr;
    if (!props2) return 1;
    VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
    props2(physical_device, &props);
    char uuid[40];  // nvidia-smi form without the prefix: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    char *out = uuid;
    for (int i = 0; i < VK_UUID_SIZE; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) *out++ = '-';
        out += sprintf(out, "%02x", id.deviceUUID[i]);
    }
    for (const char *start = cvd; start && *start;) {
        const char *end = strchr(start, ',');
        size_t len = end ? (size_t)(end - start) : strlen(start);
        if (len > 4 && !strncmp(start, "GPU-", 4) && len - 4 <= strlen(uuid) && !strncasecmp(start + 4, uuid, len - 4))
            return 1;
        start = end ? end + 1 : NULL;
    }
    return 0;
}

static VKAPI_ATTR VkResult VKAPI_CALL clamp_EnumeratePhysicalDevices(VkInstance instance, uint32_t *count,
                                                                     VkPhysicalDevice *devices) {
    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(dispatch_key(instance));
    pthread_mutex_unlock(&g_lock);
    if (!entry || !entry->enumerate) return VK_ERROR_INITIALIZATION_FAILED;
    uint32_t all = 0;
    VkResult result = entry->enumerate(instance, &all, NULL);
    if (result != VK_SUCCESS) return result;
    VkPhysicalDevice *found = calloc(all ? all : 1, sizeof(*found));
    if (!found) return VK_ERROR_OUT_OF_HOST_MEMORY;
    result = entry->enumerate(instance, &all, found);
    uint32_t kept = 0;
    for (uint32_t i = 0; result >= 0 && i < all; ++i)
        if (cuda_visible(entry, found[i])) found[kept++] = found[i];
    static int told;
    if (result >= 0 && kept < all && !told++)
        fprintf(stderr, "[vkclamp] CUDA_VISIBLE_DEVICES keeps %u of %u Vulkan devices\n", kept, all);
    if (result >= 0) {
        if (!devices) {
            *count = kept;
        } else {
            uint32_t n = *count < kept ? *count : kept;
            memcpy(devices, found, n * sizeof(*found));
            result = n < kept ? VK_INCOMPLETE : result;
            *count = n;
        }
    }
    free(found);
    return result;
}

static VkResult enumerate_groups(VkInstance instance, uint32_t *count, VkPhysicalDeviceGroupProperties *groups,
                                 int khr) {
    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(dispatch_key(instance));
    pthread_mutex_unlock(&g_lock);
    PFN_vkEnumeratePhysicalDeviceGroups next = NULL;
    if (entry) next = khr ? (entry->groups_khr ? entry->groups_khr : entry->groups)
                          : (entry->groups ? entry->groups : entry->groups_khr);
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;
    uint32_t all = 0;
    VkResult result = next(instance, &all, NULL);
    if (result != VK_SUCCESS) return result;
    VkPhysicalDeviceGroupProperties *found = calloc(all ? all : 1, sizeof(*found));
    if (!found) return VK_ERROR_OUT_OF_HOST_MEMORY;
    for (uint32_t i = 0; i < all; ++i) found[i].sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
    result = next(instance, &all, found);
    uint32_t kept = 0;
    for (uint32_t i = 0; result >= 0 && i < all; ++i) {
        uint32_t in_group = 0;
        for (uint32_t d = 0; d < found[i].physicalDeviceCount; ++d)
            if (cuda_visible(entry, found[i].physicalDevices[d]))
                found[i].physicalDevices[in_group++] = found[i].physicalDevices[d];
        if (!in_group) continue;
        found[i].physicalDeviceCount = in_group;
        if (in_group == 1) found[i].subsetAllocation = VK_FALSE;
        found[kept++] = found[i];
    }
    if (result >= 0) {
        if (!groups) {
            *count = kept;
        } else {
            uint32_t n = *count < kept ? *count : kept;
            for (uint32_t i = 0; i < n; ++i) {  // keep the caller's sType and pNext
                memcpy(groups[i].physicalDevices, found[i].physicalDevices, sizeof(found[i].physicalDevices));
                groups[i].physicalDeviceCount = found[i].physicalDeviceCount;
                groups[i].subsetAllocation = found[i].subsetAllocation;
            }
            result = n < kept ? VK_INCOMPLETE : result;
            *count = n;
        }
    }
    free(found);
    return result;
}

static VKAPI_ATTR VkResult VKAPI_CALL clamp_EnumeratePhysicalDeviceGroups(VkInstance instance, uint32_t *count,
                                                                          VkPhysicalDeviceGroupProperties *groups) {
    return enumerate_groups(instance, count, groups, 0);
}

static VKAPI_ATTR VkResult VKAPI_CALL clamp_EnumeratePhysicalDeviceGroupsKHR(
    VkInstance instance, uint32_t *count, VkPhysicalDeviceGroupProperties *groups) {
    return enumerate_groups(instance, count, groups, 1);
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
        entry->enumerate = (PFN_vkEnumeratePhysicalDevices)gipa(*instance, "vkEnumeratePhysicalDevices");
        entry->groups = (PFN_vkEnumeratePhysicalDeviceGroups)gipa(*instance, "vkEnumeratePhysicalDeviceGroups");
        entry->groups_khr =
            (PFN_vkEnumeratePhysicalDeviceGroups)gipa(*instance, "vkEnumeratePhysicalDeviceGroupsKHR");
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
    if (!strcmp(name, "vkEnumeratePhysicalDevices")) return (PFN_vkVoidFunction)clamp_EnumeratePhysicalDevices;
    if (!strcmp(name, "vkEnumeratePhysicalDeviceGroups"))
        return (PFN_vkVoidFunction)clamp_EnumeratePhysicalDeviceGroups;
    if (!strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR"))
        return (PFN_vkVoidFunction)clamp_EnumeratePhysicalDeviceGroupsKHR;
    if (instance == VK_NULL_HANDLE) return NULL;
    pthread_mutex_lock(&g_lock);
    InstanceEntry *entry = find_instance(dispatch_key(instance));
    PFN_vkGetInstanceProcAddr next = entry ? entry->gipa : NULL;
    pthread_mutex_unlock(&g_lock);
    return next ? next(instance, name) : NULL;
}
