class Context(object):
        """
        Context stores model relevant worker information
        Some fixed during load times and some set by the service
        """
        def __init__(
            self,
            model_name,
            model_dir,
            manifest,
            batch_size,
            gpu,
            mms_version,
            limit_max_image_pixels=True,
            metrics=None,
            model_yaml_config=None,
        ):
            self.model_name = model_name
            self.manifest = manifest
            self._system_properties = {
                "model_dir": model_dir,
                "gpu_id": gpu,
                "batch_size": batch_size,
                "server_name": "MMS",
                "server_version": mms_version,
                "limit_max_image_pixels": limit_max_image_pixels,
            }
            self.request_ids = None
            self.request_processor = None
            self._metrics = None
            self._limit_max_image_pixels = True
            self.metrics = metrics
            self.model_yaml_config = model_yaml_config
            self.stopping_criteria = None

        @property
        def system_properties(self):
            return self._system_properties

context = Context("test", "test", manifest={}, batch_size=4, gpu="0", mms_version=1)
# handler = Handler()
# handler.initialize(context)
