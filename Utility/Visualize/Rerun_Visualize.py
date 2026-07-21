import atexit
from functools import wraps
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from Utility.PrettyPrint import Logger
import typing as T
import pypose as pp
import numpy  as np
import torch

try:
    import rerun as rr
except ImportError:
    rr = None


T_Mode    = T.Literal["none", "rerun"]
T_Input   = T.ParamSpec("T_Input")
T_Output  = T.TypeVar("T_Output")


# NOTE: Since rerun does not ensure compatibilty between different versions,
#       We explicitly constrain the version of rerun sdk
if (rr is not None) and (rr.__version__ <= "0.20.0"):
    Logger.write("warn", f"Please re-install rerun_sdk to have version of 0.20.0. Current version is {rr.__version__}")
    rr = None

class Rerun_Visualizer:    
    func_mode: T.ClassVar[dict[str, T_Mode | T.Literal["default"]]] = dict()
    default_mode: T.ClassVar[T_Mode] = "none"
    active: T.ClassVar[bool] = False
    live_addr: T.ClassVar[str] = "127.0.0.1:9876"
    live_recording: T.ClassVar[T.Any | None] = None
    save_recording: T.ClassVar[T.Any | None] = None
    recorder_port: T.ClassVar[int | None] = None
    recorder_process: T.ClassVar[subprocess.Popen | None] = None
    managed_rrd_path: T.ClassVar[Path | None] = None
    shutdown_registered: T.ClassVar[bool] = False

    @staticmethod
    def _find_rerun_cli() -> str | None:
        rerun_cli = shutil.which("rerun")
        if rerun_cli is not None:
            return rerun_cli

        candidate = Path(sys.executable).resolve().parent / "rerun"
        if candidate.is_file():
            return str(candidate)
        return None

    @staticmethod
    def _reserve_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @staticmethod
    def _has_tcp_server(addr: str) -> bool:
        host, port_str = addr.rsplit(":", maxsplit=1)
        try:
            with socket.create_connection((host, int(port_str)), timeout=0.2):
                return True
        except OSError:
            return False

    @staticmethod
    def _iter_recordings() -> list[T.Any]:
        recordings: list[T.Any] = []
        for recording in (Rerun_Visualizer.live_recording, Rerun_Visualizer.save_recording):
            if recording is not None:
                recordings.append(recording)
        return recordings

    @staticmethod
    def _log(entity_path: str, data: object, *, static: bool = False) -> None:
        assert rr is not None
        for recording in Rerun_Visualizer._iter_recordings():
            rr.log(entity_path, data, static=static, recording=recording)

    @staticmethod
    def _send_default_blueprint() -> None:
        assert rr is not None
        import rerun.blueprint as rrb

        # Keep existing 3D/map/camera views and provide two independent IMU line-chart panes.
        bp = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/world", name="World 3D"),
                    rrb.Spatial2DView(origin="/world/macvo/cam_left", name="Camera"),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/imu/acc", contents="$origin/**", name="IMU Acc"),
                    rrb.TimeSeriesView(origin="/imu/gyro", contents="$origin/**", name="IMU Gyro"),
                ),
            ),
            auto_views=False,
            collapse_panels=False,
        )

        for recording in Rerun_Visualizer._iter_recordings():
            rr.send_blueprint(bp, make_active=True, make_default=True, recording=recording)

    @staticmethod
    def _start_managed_recorder(save_rrd: Path) -> None:
        rerun_cli = Rerun_Visualizer._find_rerun_cli()
        if rerun_cli is None:
            raise RuntimeError("Unable to find rerun CLI required for saving compact RRD recordings.")

        save_rrd.parent.mkdir(parents=True, exist_ok=True)
        recorder_port = Rerun_Visualizer._reserve_free_port()
        process = subprocess.Popen(
            [
                rerun_cli,
                "--save", str(save_rrd),
                "--serve-web",
                "--port", str(recorder_port),
                "--web-viewer-port", "0",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"rerun recorder exited immediately with code {process.returncode}.")
            try:
                with socket.create_connection(("127.0.0.1", recorder_port), timeout=0.2):
                    Rerun_Visualizer.recorder_port = recorder_port
                    Rerun_Visualizer.recorder_process = process
                    Rerun_Visualizer.managed_rrd_path = save_rrd
                    return
            except OSError:
                time.sleep(0.1)

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise RuntimeError("rerun recorder did not start listening in time.")

    @staticmethod
    def _stop_managed_recorder() -> None:
        process = Rerun_Visualizer.recorder_process
        Rerun_Visualizer.recorder_process = None
        Rerun_Visualizer.recorder_port = None
        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    @staticmethod
    def _compact_managed_rrd() -> None:
        save_rrd = Rerun_Visualizer.managed_rrd_path
        Rerun_Visualizer.managed_rrd_path = None
        if save_rrd is None or not save_rrd.exists():
            return

        rerun_cli = Rerun_Visualizer._find_rerun_cli()
        if rerun_cli is None:
            Logger.write("warn", f"Saved RRD without compaction because rerun CLI is unavailable: {save_rrd}")
            return

        with tempfile.NamedTemporaryFile(dir=save_rrd.parent, suffix=".rrd", delete=False) as temp_file:
            compact_path = Path(temp_file.name)

        result = subprocess.run(
            [rerun_cli, "rrd", "compact", str(save_rrd), "-o", str(compact_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            compact_path.unlink(missing_ok=True)
            err_msg = result.stderr.strip() if result.stderr else f"exit code {result.returncode}"
            Logger.write("warn", f"Failed to compact RRD at {save_rrd}: {err_msg}")
            return

        compact_path.replace(save_rrd)
        Logger.write("info", f"Saved compact RRD recording to {save_rrd}")
    
    @staticmethod
    def init_connect(application_id: str, save_rrd: str | Path | None = None):
        assert rr is not None, "Can't initialize rerun since rerun is not installed or have incorrect version."
        if Rerun_Visualizer.active:
            Rerun_Visualizer.shutdown()

        Rerun_Visualizer.live_recording = None
        Rerun_Visualizer.save_recording = None

        if Rerun_Visualizer._has_tcp_server(Rerun_Visualizer.live_addr):
            live_recording = rr.new_recording(f"{application_id}-live")
            rr.connect_tcp(Rerun_Visualizer.live_addr, recording=live_recording)
            Rerun_Visualizer.live_recording = live_recording
            Logger.write("info", f"Connected live rerun stream to {Rerun_Visualizer.live_addr}")
        else:
            Logger.write("warn", f"No live rerun server found at {Rerun_Visualizer.live_addr}; realtime visualization is disabled, but RRD saving will continue.")

        if save_rrd is not None:
            Rerun_Visualizer._start_managed_recorder(Path(save_rrd))
            save_recording = rr.new_recording(f"{application_id}-save")
            rr.connect_tcp(f"127.0.0.1:{Rerun_Visualizer.recorder_port}", recording=save_recording)
            Rerun_Visualizer.save_recording = save_recording
            Logger.write("info", f"Saving rerun stream to {save_rrd}")

        if not Rerun_Visualizer._iter_recordings():
            raise RuntimeError("Unable to initialize rerun output: no live rerun server is available and no save recorder was created.")

        Rerun_Visualizer._log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)
        Rerun_Visualizer.active = True
        Rerun_Visualizer._send_default_blueprint()
        if not Rerun_Visualizer.shutdown_registered:
            atexit.register(Rerun_Visualizer.shutdown)
            Rerun_Visualizer.shutdown_registered = True
    
    @staticmethod
    def init_save(application_id: str, save_rrd: str):
        Rerun_Visualizer.init_connect(application_id, save_rrd)

    @staticmethod
    def shutdown() -> None:
        if rr is not None and Rerun_Visualizer.active:
            for recording in Rerun_Visualizer._iter_recordings():
                try:
                    rr.disconnect(recording=recording)
                except Exception as exc:
                    Logger.write("warn", f"Failed to disconnect rerun stream cleanly: {exc}")

        Rerun_Visualizer.active = False
        Rerun_Visualizer.live_recording = None
        Rerun_Visualizer.save_recording = None
        Rerun_Visualizer._stop_managed_recorder()
        Rerun_Visualizer._compact_managed_rrd()

    @staticmethod
    def set_time_sequence(timeline: str, sequence: int) -> None:
        if (rr is None) or (not Rerun_Visualizer.active):
            return
        for recording in Rerun_Visualizer._iter_recordings():
            rr.set_time_sequence(timeline, sequence, recording=recording)

    @staticmethod
    def set_fn_mode(func: T.Callable[T.Concatenate[str, T_Input], None], mode: T_Mode | T.Literal["default"]):
        Rerun_Visualizer.func_mode[func.__name__] = mode

    @staticmethod
    def get_fn_mode(func: T.Callable[T.Concatenate[str, T_Input], None]) -> T_Mode:
        assert func.__name__ in Rerun_Visualizer.func_mode
        func_mode = Rerun_Visualizer.func_mode[func.__name__]
        if func_mode == "default": return Rerun_Visualizer.default_mode
        return func_mode
    
    @staticmethod
    def register(func: T.Callable[T.Concatenate[str, T_Input], None]) -> T.Callable[T.Concatenate[str, T_Input], None]:
        @wraps(func)
        def implement(rerun_path: str, *args: T_Input.args, **kwargs: T_Input.kwargs) -> None:
            if func.__name__ not in Rerun_Visualizer.func_mode:
                Rerun_Visualizer.func_mode[func.__name__] = "default"
            
            func_mode = Rerun_Visualizer.get_fn_mode(func)
            if not Rerun_Visualizer.active:
                return None
            match func_mode:
                case "none": return None
                case "rerun": func(rerun_path, *args, **kwargs)
        return implement

    @register
    @staticmethod
    def log_trajectory(rerun_path: str, trajectory: pp.LieTensor | torch.Tensor, **kwargs):
        assert rr is not None
        if not isinstance(trajectory, pp.LieTensor):
            trajectory = pp.SE3(trajectory)
        
        position = trajectory.translation().detach().cpu().numpy()
        from_pos = position[:-1]
        to_pos = position[1:]
        Rerun_Visualizer._log(rerun_path, rr.LineStrips3D(np.stack([from_pos, to_pos], axis=1), **kwargs))

    @register
    @staticmethod
    def log_camera(rerun_path: str, pose: pp.LieTensor | torch.Tensor, K: torch.Tensor, **kwargs):
        assert rr is not None
        cx = K[0][2].item()
        cy = K[1][2].item()
        
        if not isinstance(pose, pp.LieTensor):
            pose = pp.SE3(pose)
        frame_position = pose.translation().detach().cpu().numpy()
        frame_rotation = pose.rotation().detach().cpu().numpy()

        Rerun_Visualizer._log(
            "/".join(rerun_path.split("/")[:-1]),
            rr.Transform3D(
                translation=frame_position,
                rotation=rr.datatypes.Quaternion(xyzw=frame_rotation),
            ),
        )
        Rerun_Visualizer._log(
            rerun_path,
            rr.Pinhole(
                resolution=[cx * 2, cy * 2],
                image_from_camera=K.detach().cpu().numpy(),
                camera_xyz=rr.ViewCoordinates.FRD,
                image_plane_distance=0.25
            ),
        )

    @register
    @staticmethod
    def log_points(rerun_path: str, position: torch.Tensor, color: torch.Tensor | None, cov_Tw: torch.Tensor | None, cov_mode: T.Literal["none", "axis", "sphere", "color"]="sphere"):
        assert rr is not None
        Rerun_Visualizer._log(
            rerun_path, 
            rr.Points3D(positions=position, colors=color.detach().cpu().numpy() if (color is not None) else None)
        )
        
        match cov_Tw, cov_mode:
            case None, _: return
            case _, "none": return
            case _, "axis":
                eigen_val, eigen_vec = torch.linalg.eig(cov_Tw)
                eigen_val, eigen_vec = eigen_val.real, eigen_vec.real

                delta = position.repeat(1, 3, 1).reshape(-1, 3)
                eigen_vec_Tw = eigen_vec.transpose(-1, -2).reshape(-1, 3)
                eigen_val = eigen_val.unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3)
                eigen_vec_Tw = eigen_vec_Tw * eigen_val.sqrt()
                eigen_vec_Tw_a = delta + .1 * eigen_vec_Tw
                eigen_vec_Tw_b = delta - .1 * eigen_vec_Tw
                Rerun_Visualizer._log(
                    rerun_path + "/cov",
                    rr.LineStrips3D(
                        torch.stack([eigen_vec_Tw_a, eigen_vec_Tw_b], dim=1).numpy(),
                        radii=[0.003],
                        colors=color.unsqueeze(0).repeat(3, 1, 1).reshape(-1, 3) if (color is not None) else None
                    ),
                )
            case _, "sphere":
                radii  = (cov_Tw.det().sqrt() * 1e2).clamp(min=0.03, max=0.5)
                Rerun_Visualizer._log(
                    rerun_path + "/cov", 
                    rr.Points3D(positions=position, colors=color.detach().cpu().numpy() if (color is not None) else None,
                                radii=radii)
                )
            case _, "color":
                import matplotlib.pyplot as plt
                from matplotlib.colors import Normalize
                cov_value = cov_Tw.det()
                cov_det_normalized = Normalize(vmin=0, vmax=cov_value.quantile(0.99).item())(cov_value)
                colormap = plt.cm.plasma    #type: ignore
                c = colormap(cov_det_normalized)[..., :3]        
                Rerun_Visualizer._log(rerun_path + "/cov", rr.Points3D(position, colors=c))

    @register
    @staticmethod
    def log_image(rerun_path: str, image: torch.Tensor | np.ndarray):
        assert rr is not None
        if isinstance(image, torch.Tensor): np_image = image.cpu().numpy()
        else: np_image = image
        
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)
        Rerun_Visualizer._log(rerun_path, rr.Image(np_image).compress())

    @register
    @staticmethod
    def log_scalar(rerun_path: str, value: float | int):
        assert rr is not None
        Rerun_Visualizer._log(rerun_path, rr.Scalar(float(value)))
