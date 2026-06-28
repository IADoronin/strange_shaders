// topology_native — нативный (Rust + wgpu / Metal) raymarcher топологических
// пространств + оптика «заторенной вселенной». Один полноэкранный треугольник,
// вся логика — во фрагментном WGSL-шейдере (src/shader.wgsl).
use std::sync::Arc;
use std::time::Instant;
use muda::{CheckMenuItem, Menu, MenuEvent, MenuId, PredefinedMenuItem, Submenu};
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    a: [f32; 4], // res.x, res.y, time, cell
    b: [f32; 4], // mouse.x, mouse.y, exposure, star_lum
    c: [f32; 4], // mode, pad, pad, pad
}

struct State {
    mouse: [f32; 2],
    mode: i32,
    cell: f32,
    exposure: f32,
    star_lum: f32,
    speed: f32,
    paused: bool,
    sim_time: f32,
    last: Instant,
    width: f32,
    height: f32,
}

const MODE_NAMES: [&str; 6] = [
    "T³ (тор)",
    "Зеркальный орбифолд",
    "Скрученный T³",
    "Звезда+планета (T³)",
    "Обычное R³",
    "Заторенная вселенная",
];

fn title(s: &State) -> String {
    format!(
        "topology_native | режим {}·{} | период L={:.1} | exposure={:.2} | star_lum={:.2}",
        s.mode + 1,
        MODE_NAMES[s.mode as usize],
        s.cell,
        s.exposure,
        s.star_lum
    )
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("topology_native")
            .with_inner_size(LogicalSize::new(1100.0, 700.0))
            .build(&event_loop)
            .unwrap(),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    println!("adapter: {:?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats[0];
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // --- pipeline ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let ubo = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ubo"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: ubo.as_entire_binding(),
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipe"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // --- меню (нативная строка меню macOS) с выбором топологии ---
    let menu = Menu::new();
    #[cfg(target_os = "macos")]
    {
        let app_menu = Submenu::new("Topology Native", true);
        let _ = app_menu.append(&PredefinedMenuItem::about(Some("Topology Native"), None));
        let _ = app_menu.append(&PredefinedMenuItem::separator());
        let _ = app_menu.append(&PredefinedMenuItem::quit(None));
        let _ = menu.append(&app_menu);
    }
    let topo_menu = Submenu::new("Топология", true);
    let mut topo_items: Vec<CheckMenuItem> = Vec::new();
    for i in 0..6usize {
        let label = format!("{} · {}", i + 1, MODE_NAMES[i]);
        let it = CheckMenuItem::new(label, true, i == 0, None);
        let _ = topo_menu.append(&it);
        topo_items.push(it);
    }
    let _ = menu.append(&topo_menu);
    #[cfg(target_os = "macos")]
    menu.init_for_nsapp();
    let topo_ids: Vec<MenuId> = topo_items.iter().map(|x| x.id().clone()).collect();

    let mut state = State {
        mouse: [0.0, 0.0],
        mode: 0,
        cell: 8.0,
        exposure: 1.2,
        star_lum: 12.0,
        speed: 1.0,
        paused: false,
        sim_time: 0.0,
        last: Instant::now(),
        width: size.width.max(1) as f32,
        height: size.height.max(1) as f32,
    };
    window.set_title(&title(&state));
    print_controls();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            // клики по меню «Топология»
            while let Ok(ev) = MenuEvent::receiver().try_recv() {
                if let Some(i) = topo_ids.iter().position(|id| id == &ev.id) {
                    state.mode = i as i32;
                    for (j, it) in topo_items.iter().enumerate() {
                        it.set_checked(j as i32 == state.mode);
                    }
                    window.set_title(&title(&state));
                }
            }

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(new) => {
                        config.width = new.width.max(1);
                        config.height = new.height.max(1);
                        state.width = config.width as f32;
                        state.height = config.height as f32;
                        surface.configure(&device, &config);
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        state.mouse[0] = (position.x as f32 / state.width) * 2.0 - 1.0;
                        state.mouse[1] = -((position.y as f32 / state.height) * 2.0 - 1.0);
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let dy = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                        };
                        let f = if dy > 0.0 { 1.15 } else { 0.87 };
                        state.speed = (state.speed * f).clamp(0.05, 8.0);
                    }
                    WindowEvent::KeyboardInput { event: ke, .. } => {
                        if ke.state == ElementState::Pressed {
                            let mut changed = true;
                            match &ke.logical_key {
                                Key::Character(s) => match s.as_str() {
                                    "1" => state.mode = 0,
                                    "2" => state.mode = 1,
                                    "3" => state.mode = 2,
                                    "4" => state.mode = 3,
                                    "5" => state.mode = 4,
                                    "6" => state.mode = 5,
                                    "]" => {
                                        let s = (state.cell * 0.1).round().max(1.0);
                                        state.cell = (state.cell + s).min(240.0);
                                    }
                                    "[" => {
                                        let s = (state.cell * 0.1).round().max(1.0);
                                        state.cell = (state.cell - s).max(4.0);
                                    }
                                    "=" | "+" => state.exposure = (state.exposure * 1.2).min(40.0),
                                    "-" | "_" => state.exposure = (state.exposure / 1.2).max(0.02),
                                    "." | ">" => state.star_lum = (state.star_lum * 1.25).min(60.0),
                                    "," | "<" => state.star_lum = (state.star_lum / 1.25).max(0.05),
                                    _ => changed = false,
                                },
                                Key::Named(NamedKey::Space) => state.paused = !state.paused,
                                Key::Named(NamedKey::Escape) => elwt.exit(),
                                _ => changed = false,
                            }
                            if changed {
                                window.set_title(&title(&state));
                                for (j, it) in topo_items.iter().enumerate() {
                                    it.set_checked(j as i32 == state.mode);
                                }
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = (now - state.last).as_secs_f32();
                        state.last = now;
                        if !state.paused {
                            state.sim_time += dt * state.speed;
                        }

                        let uni = Uniforms {
                            a: [state.width, state.height, state.sim_time, state.cell],
                            b: [state.mouse[0], state.mouse[1], state.exposure, state.star_lum],
                            c: [state.mode as f32, 0.0, 0.0, 0.0],
                        };
                        queue.write_buffer(&ubo, 0, bytemuck::bytes_of(&uni));

                        let frame = match surface.get_current_texture() {
                            Ok(f) => f,
                            Err(_) => {
                                surface.configure(&device, &config);
                                return;
                            }
                        };
                        let view = frame.texture.create_view(&Default::default());
                        let mut enc = device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                        {
                            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("raymarch"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.0,
                                            g: 0.0,
                                            b: 0.0,
                                            a: 1.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            rp.set_pipeline(&pipeline);
                            rp.set_bind_group(0, &bind_group, &[]);
                            rp.draw(0..3, 0..1);
                        }
                        queue.submit(Some(enc.finish()));
                        frame.present();
                        window.request_redraw();
                    }
                    _ => {}
                },
                _ => {}
            }
        })
        .unwrap();
}

fn print_controls() {
    println!("--- управление ---");
    println!("1..6  режимы (6 = заторенная вселенная)");
    println!("[ ]   период заворота L");
    println!("- =   экспозиция (режим 6)");
    println!(", .   светимость звезды (режим 6)");
    println!("мышь  осмотр · колесо  скорость · Space  пауза · Esc  выход");
}
