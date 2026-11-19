# goroutine

goroutine 是 Go 运行时（runtime）管理的轻量级线程，由 `go` 关键字启动。它不是操作系统线程，而是用户态的调度单元。

## 调度与开销

初始栈只有约 2KB，可按需动态增长（对比 OS 线程通常固定 1MB+）。调度由 GMP 模型完成，切换不涉及系统调用，开销远小于 OS 线程。因此单机启动上万乃至百万个 goroutine 是可行的。

## 适用场景

适合高并发 I/O 与并行任务。但 CPU 密集任务受 GOMAXPROCS 限制，盲目加 goroutine 不会提升吞吐，反而增加调度开销。

# channel

channel 是 goroutine 之间通信与同步的管道，类型安全且线程安全，是 Go「不要通过共享内存来通信，而要通过通信来共享内存」哲学的核心载体。

## 缓冲与同步

分为无缓冲（unbuffered）和有缓冲（buffered）两种。无缓冲 channel 的发送与接收必须同时就绪，天然形成同步点；有缓冲 channel 在缓冲区未满/未空时不阻塞。

## 常见操作

支持 `for-range` 遍历、`close()` 关闭、`select` 多路复用。向已关闭的 channel 发送会 panic，从已关闭的 channel 接收会立即返回零值，可用 `v, ok := <-ch` 判断是否关闭。

# sync.Mutex

sync.Mutex 是互斥锁，用于保护临界区，保证同一时刻只有一个 goroutine 访问共享资源。

## 基本用法

`Lock()` 加锁，`Unlock()` 解锁，必须成对出现，通常用 `defer mu.Unlock()` 保证异常路径也能释放。忘记解锁是死锁的常见来源。

## 饥饿模式

Go 1.18+ 的 Mutex 支持 starvation（饥饿）模式：当等待者等待超过 1ms，锁进入饥饿模式，按 FIFO 交给等待最久的 goroutine，避免公平性问题。

# interface

interface 是 Go 的抽象类型，定义一组方法签名。它是 Go 实现多态与依赖倒置的核心。

## 隐式实现

任何实现了接口全部方法的类型都隐式满足该接口（鸭子类型），无需像 Java 那样显式 `implements`。这让定义方与实现方解耦。

## 底层结构

空接口 `interface{}`（Go 1.18+ 可用 `any`）可持有任意类型值，底层由 type + value 两部分组成（iface / eface）。带方法的接口用 iface，空接口用 eface。

# defer

defer 把函数调用推迟到当前函数返回前执行，常用于资源释放、锁释放、panic 恢复。

## 执行顺序

多个 defer 按后进先出（LIFO）顺序执行。defer 语句的参数在注册时就求值，但函数体在返回时才真正执行——这是最容易踩坑的点。

## 与 recover 配合

defer 中调用 `recover()` 可以捕获 panic，实现类似异常处理的机制。recover 只在 defer 函数中直接调用才有效。

# context

context 是 Go 中跨 API 边界传递取消信号、超时和请求级元数据的标准方式，是构建可取消、可超时服务的基础设施。

## 取消传播

`context.WithCancel` 返回的 Context 在父级取消或手动调用 cancel() 时，其 Done() channel 会关闭，所有下游 goroutine 可据此退出，实现级联取消。

## 超时控制

`context.WithTimeout` 和 `context.WithDeadline` 用于设置超时。标准库的 http、database/sql 等都接受 ctx 参数，是编写可中断服务的惯例。

# GMP 调度模型

GMP 是 Go 运行时的调度模型：G（goroutine）、M（machine，OS 线程）、P（processor，调度上下文）。

## 三者关系

G 是待执行任务，M 是执行任务的 OS 线程，P 是调度器上下文，持有可运行 G 的本地队列。M 必须绑定 P 才能执行 G。P 的数量由 GOMAXPROCS 决定，默认等于 CPU 核数。

## work stealing

当某个 P 的本地队列为空时，会从其他 P 的队列「偷」一半 G 来执行，保证各核负载均衡。这是 Go 调度器高效的关键之一。
